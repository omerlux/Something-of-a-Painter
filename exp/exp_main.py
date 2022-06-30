import os
import wandb
import shutil
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from wandb.keras import WandbCallback

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import CycleGan
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import discriminator_loss, generator_loss, calc_cycle_loss, identity_loss, \
    load_inception_model, MiFID, calculate_activation_statistics_mod
from utils.tools import display_samples, display_augmented_samples, display_generated_samples, predict_and_save, \
    LogCallback, ClearMemory

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args, logger):
        super(Exp_Main, self).__init__(args, logger)

    def _build_model(self):
        model_dict = {
            'CycleGan': CycleGan,
            'ResCycleGan': CycleGan
        }

        model = model_dict[self.args.model].Model(self.args)
        self.logger.info("\nGenerator:\n")
        model.m_gen.summary(print_fn=self.logger.info)
        self.logger.info("\n\nDiscriminator:\n")
        model.m_disc.summary(print_fn=self.logger.info)
        return model

    def _get_data(self):
        gan_ds, (n_monet, monet_ds), (n_photo, photo_ds) = data_provider(self.args, self.logger)
        self.n_monet = n_monet
        self.n_photo = n_photo
        self.gan_ds = gan_ds
        self.monet_ds = monet_ds
        self.photo_ds = photo_ds
        self.steps_per_epoch = max(self.n_monet, self.n_photo) // 4 if self.args.steps_per_epoch == -1 \
                                   else self.args.steps_per_epoch
        return gan_ds, monet_ds, photo_ds

    def _select_optimizer(self):
        # Create generators
        monet_generator_optimizer = optimizers.Adam(self.args.learning_rate, beta_1=0.5)
        photo_generator_optimizer = optimizers.Adam(self.args.learning_rate, beta_1=0.5)

        # Create discriminators
        monet_discriminator_optimizer = optimizers.Adam(self.args.learning_rate, beta_1=0.5)
        photo_discriminator_optimizer = optimizers.Adam(self.args.learning_rate, beta_1=0.5)

        return monet_generator_optimizer, photo_generator_optimizer, \
               monet_discriminator_optimizer, photo_discriminator_optimizer

    def train(self, setting):
        self.logger.info("| Loading data '{}'".format(self.args.data))
        gan_ds, monet_ds, photo_ds = self._get_data()
        display_samples(path=self.args.save, name="Monet", ds=monet_ds.batch(1), row=4, col=6)
        display_samples(path=self.args.save, name="Photo", ds=photo_ds.batch(1), row=4, col=6)
        display_augmented_samples(path=self.args.save, name="Monet_Diff-{}".format(self.args.diffaugment),
                                  ds=monet_ds.batch(1), num_images=4)
        display_augmented_samples(path=self.args.save, name="Photo_Diff-{}".format(self.args.diffaugment),
                                  ds=photo_ds.batch(1), num_images=4)
        display_augmented_samples(path=self.args.save, name="Monet_Data-{}".format(self.args.ds_augment),
                                  ds=monet_ds.batch(1), num_images=4, diffaugment=False)
        display_augmented_samples(path=self.args.save, name="Photo_Data-{}".format(self.args.ds_augment),
                                  ds=photo_ds.batch(1), num_images=4, diffaugment=False)

        chkpnt_exists = [f for f in os.listdir(self.args.save) if
                         not (('log' in f) or ('scripts' in f) or ('images' in f) or ('examples' in f))]
        if len(chkpnt_exists) != 0:
            setting = chkpnt_exists[0]
        self.chkpath = os.path.join(self.args.save, setting, self.args.checkpoints)
        if not os.path.exists(self.chkpath):
            os.makedirs(self.chkpath)

        monet_gn_opt, photo_gn_opt, monet_ds_opt, photo_ds_opt = self._select_optimizer()
        self.model.compile(m_gen_optimizer=monet_gn_opt,
                           p_gen_optimizer=photo_gn_opt,
                           m_disc_optimizer=monet_ds_opt,
                           p_disc_optimizer=photo_ds_opt,
                           gen_loss_fn=generator_loss,
                           disc_loss_fn=discriminator_loss,
                           cycle_loss_fn=calc_cycle_loss,
                           identity_loss_fn=identity_loss,
                           diffaugment=self.args.diffaugment)

        if len(chkpnt_exists) != 0:
            self.model.load_weights(os.path.join(self.chkpath, 'cp.ckpt'))
        else:
            self.model.save_weights(os.path.join(self.chkpath, 'cp.ckpt'))

        callbacks = [LogCallback(self.logger, self.args.log_interval),
                     ClearMemory(self.logger),
                     keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.chkpath, 'cp.ckpt'),
                                                     save_weights_only=True,
                                                     verbose=1)]

        if self.args.wandb:
            callbacks.append(WandbCallback(log_batch_frequency=self.args.log_interval))  # 10

        self.history = self.model.fit(
            gan_ds,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.args.train_epochs,
            verbose=1,
            callbacks=callbacks
        )

        # Save the whole model
        self.model.compute_output_shape(input_shape=(None, self.args.height, self.args.width, self.args.channels))
        self.monet_generator = self.model.m_gen
        keras.Model.save(self.monet_generator, os.path.join(self.chkpath, "model_m_gen"), save_format='tf')
        # tf.saved_model.save(self.model, os.path.join(self.chkpath, "model"))

        # calculate the FID / MiFID scores:
        inception_model = load_inception_model()
        FID_mu2, FID_sigma2, FID_features2 = calculate_activation_statistics_mod(monet_ds.batch(self.args.batch_size),
                                                                                 inception_model)
        MiFID_score = MiFID(images=photo_ds.batch(self.args.batch_size), gen_model=self.monet_generator,
                            inception_model=inception_model, mu2=FID_mu2, sigma2=FID_sigma2, features2=FID_features2)
        self.logger.info("=" * 80)
        self.logger.info("| MiFID Score: {:3.5f}".format(MiFID_score))
        self.logger.info("=" * 80)
        if self.args.wandb:
            wandb.log({"MiFID":  MiFID_score})

        display_generated_samples(
            path=self.args.save,
            ds=photo_ds.batch(1),
            model=self.monet_generator,
            n_samples=30
        )

        return self.model

    def predict(self, setting):
        self.chkpath = os.path.join(self.args.save, setting, self.args.checkpoints)
        loaded_model = tf.keras.models.load_model(os.path.join(self.chkpath, "model_m_gen"))
        # self.model.load_weights(os.path.join(self.chkpath, 'cp.ckpt'))
        _, _, photo_ds = self._get_data()

        predict_and_save(
            path=self.args.save,
            input_ds=photo_ds.batch(1),
            generator_model=loaded_model  # Monet Generator
        )
        images_path = os.path.join(self.args.save, 'images')
        shutil.make_archive(images_path, 'zip', self.args.save)
        self.logger.info('| Generated samples: {}'.format(
            len([name for name in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, name))])
        ))
