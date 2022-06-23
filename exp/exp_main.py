import os
import wandb
import shutil
import warnings
import numpy as np
from tensorflow import keras
from tensorflow.keras import optimizers
from wandb.keras import WandbCallback

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import CycleGan
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import discriminator_loss, generator_loss, calc_cycle_loss, identity_loss
from utils.tools import display_samples, display_augmented_samples, display_generated_samples, predict_and_save

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
        display_samples(path=self.args.save, name="monet", ds=monet_ds.batch(1), row=4, col=6)
        display_samples(path=self.args.save, name="photo", ds=photo_ds.batch(1), row=4, col=6)
        display_augmented_samples(path=self.args.save, name="monet", ds=monet_ds.batch(1), num_images=4)
        display_augmented_samples(path=self.args.save, name="photo", ds=monet_ds.batch(1), num_images=4)

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
                           augment=self.args.augment)
        # TODO: save checkpoints

        if self.args.wandb:
            self.history = self.model.fit(
                gan_ds,
                steps_per_epoch=(self.n_monet // self.args.batch_size),
                epochs=self.args.train_epochs,
                verbose=1,
                callbacks=[WandbCallback(
                    log_batch_frequency=self.args.log_interval,  # 10
                    # log_evaluation_frequency=1
                ),
                    keras.callbacks.ModelCheckpoint(filepath=self.chkpath,
                                                    save_weights_only=True,
                                                    verbose=1)
                ]
            )
        else:
            self.history = self.model.fit(
                gan_ds,
                steps_per_epoch=(self.n_monet // self.args.batch_size),
                epochs=self.args.train_epochs,
                verbose=1,
                callbacks=[
                    keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.chkpath, 'cp.ckpt'),
                                                    save_weights_only=True,
                                                    verbose=1)
                ]
            )

        display_generated_samples(
            path=self.args.save,
            ds=photo_ds.batch(1),
            model=self.model.m_gen,
            n_samples=10
        )

        return self.model

    def predict(self, setting):
        self.chkpath = os.path.join(self.args.save, setting, self.args.checkpoints)
        self.model.load_weights(os.path.join(self.chkpath, 'cp.ckpt'))

        predict_and_save(
            path=self.args.save,
            input_ds=self.photo_ds.batch(1),
            generator_model=self.model.m_gen
        )
        images_path = os.path.join(self.args.save, 'generated')
        shutil.make_archive(images_path, 'zip')
        self.logger.info('| Generated samples: {}'.format(
            len([name for name in images_path if os.path.isfile(os.path.join(images_path, name))])
        ))
