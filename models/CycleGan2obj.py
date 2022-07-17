import tensorflow as tf
import tensorflow.keras as keras
import layers.Discriminator as Discriminator
import layers.Generator as Generator
from layers.Augmentations import DiffAugment, DataAugment
import wandb


class Model(keras.Model):
    def __init__(self, args, lambda_cycle=10, ):
        super(Model, self).__init__()
        self.height = args.height
        self.width = args.width
        self.channels = args.channels
        self.out_channels = args.out_channels
        self.ds_augment = args.ds_augment
        self.cycle_noise = args.cycle_noise     # represents the std of the noise - 0 for none
        self.dsaug_layer = DataAugment(args.height, args.width, args.channels)
        self.m_gen = Generator.generator_fn(args.height, args.width, args.channels, args.out_channels)
        self.p_gen = Generator.generator_fn(args.height, args.width, args.channels, args.out_channels)
        self.m_disc = Discriminator.discriminator_fn_prehead(args.height, args.width, args.channels)
        self.p_disc = Discriminator.discriminator_fn(args.height, args.width, args.channels)
        self.dhead1 = Discriminator.discriminator_head()        # Head for BCE
        self.dhead2 = Discriminator.discriminator_head()        # Head for Hinge Loss
        self.wandb = args.wandb
        self.lambda_cycle = lambda_cycle
        self.build((None,) + (self.height, self.width, self.channels))

    def compile(self,
                m_gen_optimizer,
                p_gen_optimizer,
                m_disc_optimizer,
                p_disc_optimizer,
                gen_loss_fn1,
                gen_loss_fn2,
                disc_loss_fn1,
                disc_loss_fn2,
                cycle_loss_fn,
                identity_loss_fn,
                diffaugment
                ):
        super(Model, self).compile()
        self.m_gen_optimizer = m_gen_optimizer
        self.p_gen_optimizer = p_gen_optimizer
        self.m_disc_optimizer = m_disc_optimizer
        self.p_disc_optimizer = p_disc_optimizer
        self.gen_loss_fn1 = gen_loss_fn1
        self.gen_loss_fn2 = gen_loss_fn2
        self.disc_loss_fn1 = disc_loss_fn1
        self.disc_loss_fn2 = disc_loss_fn2
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn
        self.diffaugment = diffaugment

    def call(self, input, training=False):
        return self.m_gen(input, training=training)

    def train_step(self, batch_data):
        real_monet, real_photo = batch_data
        # dataset augmentation layer
        if self.ds_augment:
            real_monet = self.dsaug_layer(real_monet)
            real_photo = self.dsaug_layer(real_photo)

        batch_size = tf.shape(real_monet)[0]
        with tf.GradientTape(persistent=True) as tape:
            # photo to monet back to photo
            fake_monet = self.m_gen(real_photo, training=True)
            if self.cycle_noise:
                fake_monet += tf.random.normal(tf.shape(fake_monet), mean=0, stddev=self.cycle_noise)
            cycled_photo = self.p_gen(fake_monet, training=True)

            # monet to photo back to monet
            fake_photo = self.p_gen(real_monet, training=True)
            if self.cycle_noise:
                fake_photo += tf.random.normal(tf.shape(fake_photo), mean=0, stddev=self.cycle_noise)
            cycled_monet = self.m_gen(fake_photo, training=True)

            # generating itself
            same_monet = self.m_gen(real_monet, training=True)
            same_photo = self.p_gen(real_photo, training=True)

            # "Diffaugment": Augmentation is done before the discriminator
            if len(self.diffaugment) != 0:
                both_monet = tf.concat([real_monet, fake_monet], axis=0)
                aug_monet = DiffAugment(both_monet, self.diffaugment)
                real_monet = aug_monet[:batch_size]  # AUGMENTED!
                fake_monet = aug_monet[batch_size:]  # AUGMENTED!

            # ------------------------------------------------------------------------------------
            # HEAD 1:
            # discriminator used to check, inputing real images
            disc_real_monet1 = self.dhead1(self.m_disc(real_monet, training=True), training=True)
            # discriminator used to check, inputing fake images
            disc_fake_monet1 = self.dhead1(self.m_disc(fake_monet, training=True), training=True)
            # evaluates generator loss
            monet_gen_loss1 = self.gen_loss_fn1(disc_fake_monet1)
            # evaluates discriminator loss
            monet_disc_loss_head1 = self.disc_loss_fn1(disc_real_monet1, disc_fake_monet1)

            # HEAD 2:
            # discriminator used to check, inputing real images
            disc_real_monet2 = self.dhead2(self.m_disc(real_monet, training=True), training=True)
            # discriminator used to check, inputing fake images
            disc_fake_monet2 = self.dhead2(self.m_disc(fake_monet, training=True), training=True)
            # evaluates generator loss
            monet_gen_loss2 = self.gen_loss_fn1(disc_fake_monet2)
            # evaluates discriminator loss
            monet_disc_loss_head2 = self.disc_loss_fn1(disc_real_monet2, disc_fake_monet2)

            # adding head1 and head2 results:
            monet_gen_loss = (monet_gen_loss1 + monet_gen_loss2) * 0.4
            monet_disc_loss = monet_disc_loss_head1 + monet_disc_loss_head2
            # ------------------------------------------------------------------------------------

            # discriminator used to check, inputing real images
            disc_real_photo = self.p_disc(real_photo, training=True)
            # discriminator used to check, inputing fake images
            disc_fake_photo = self.p_disc(fake_photo, training=True)
            # evaluates generator loss
            photo_gen_loss = self.gen_loss_fn1(disc_fake_photo)
            # evaluates discriminator loss
            photo_disc_loss = self.disc_loss_fn1(disc_real_photo, disc_fake_photo)

            # evaluates total cycle consistency loss
            total_cycle_loss = self.cycle_loss_fn(real_monet, cycled_monet, self.lambda_cycle) + \
                               self.cycle_loss_fn(real_photo, cycled_photo, self.lambda_cycle)

            # evaluates total generator loss
            total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(real_monet, same_monet,
                                                                                             self.lambda_cycle)
            total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(real_photo, same_photo,
                                                                                             self.lambda_cycle)

        # Calculate the gradients for generator and discriminator
        monet_generator_gradients = tape.gradient(total_monet_gen_loss,
                                                  self.m_gen.trainable_variables)
        photo_generator_gradients = tape.gradient(total_photo_gen_loss,
                                                  self.p_gen.trainable_variables)

        monet_discriminator_gradients = tape.gradient(monet_disc_loss,
                                                      self.m_disc.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_disc_loss,
                                                      self.p_disc.trainable_variables)

        # Apply the gradients to the optimizer
        self.m_gen_optimizer.apply_gradients(zip(monet_generator_gradients,
                                                 self.m_gen.trainable_variables))

        self.p_gen_optimizer.apply_gradients(zip(photo_generator_gradients,
                                                 self.p_gen.trainable_variables))

        self.m_disc_optimizer.apply_gradients(zip(monet_discriminator_gradients,
                                                  self.m_disc.trainable_variables))

        self.p_disc_optimizer.apply_gradients(zip(photo_discriminator_gradients,
                                                  self.p_disc.trainable_variables))

        return {
            'monet_gen_loss': total_monet_gen_loss,
            'photo_gen_loss': total_photo_gen_loss,
            'monet_disc_loss': monet_disc_loss,
            'photo_disc_loss': photo_disc_loss,
            'total_cycle_loss': total_cycle_loss,
            "monet_disc_loss_head1": monet_disc_loss_head1,
            "monet_disc_loss_head2": monet_disc_loss_head2,
            "disc_real_monet1": disc_real_monet1,
            "disc_fake_monet1": disc_fake_monet1,
            "disc_real_monet2": disc_real_monet2,
            "disc_fake_monet2": disc_fake_monet2,
        }
