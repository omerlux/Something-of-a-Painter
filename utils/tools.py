import os
import gc
import PIL
import torch
import shutil
import resource
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
from layers.Augmentations import DiffAugment, DataAugment

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, logger):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        # lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
        lr_adjust = {epoch: args.learning_rate * (0.95 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        lr_adjust = {5: args.learning_rate * 0.5,
                     10: args.learning_rate * 0.25}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logger.info('| * Updating learning rate to {:2.2e}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, logger=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.logger = logger

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.logger.info(f'| * EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.logger.info(f'| * Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.grid()
    plt.savefig(name, bbox_inches='tight')


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path.split('/')[0]):
        os.mkdir(path.split('/')[0])
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'images'))
        os.mkdir(os.path.join(path, 'examples'))
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def display_samples(path, name, ds, row, col):
    ds_iter = iter(ds)
    fig = plt.figure(figsize=(15, int(15 * row / col)))
    for j in range(row * col):
        example_sample = next(ds_iter)
        plt.subplot(row, col, j + 1)
        plt.axis('off')
        plt.imshow(example_sample[0] * 0.5 + 0.5)
    path = os.path.join(path, "examples", f"{name}.png")
    fig.suptitle(f"{name}")
    plt.savefig(path)
    plt.show()


def display_augmented_samples(path, name, ds, num_images=1, diffaugment=True):
    ds_iter = iter(ds)
    dsaug_layer = DataAugment()

    fig = plt.figure(figsize=(5 * num_images, 10))
    for j in range(num_images):
        example_sample = next(ds_iter)
        img = example_sample * 0.5 + 0.5
        if diffaugment:
            x = DiffAugment(img)
        else:
            x = dsaug_layer(img)
        ax = plt.subplot(2, num_images, j + 1)
        plt.axis('off')
        ax.set_title("Origin")
        ax.imshow(img[0])
        ax = plt.subplot(2, num_images, (j + num_images) + 1)
        plt.axis('off')
        ax.set_title("Augmented")
        ax.imshow(x[0])
    path = os.path.join(path, "examples", f"{name.split('-')[0]}_Augmented.png")
    fig.suptitle(f"{name} Augmentations")
    plt.savefig(path)
    plt.show()


def display_generated_samples(path, ds, model, n_samples):
    ds_iter = iter(ds)
    for n_sample in range(n_samples):
        example_sample = next(ds_iter)
        generated_sample = model.predict(example_sample)

        plt.subplot(121)
        plt.title("Input image")
        plt.imshow(example_sample[0] * 0.5 + 0.5)
        plt.axis('off')

        plt.subplot(122)
        plt.title("Generated image")
        plt.imshow(generated_sample[0] * 0.5 + 0.5)
        plt.axis('off')

        path_tmp = os.path.join(path, "examples", "{0:02d}.png".format(n_sample))
        plt.savefig(path_tmp)

        plt.show()


def predict_and_save(path, input_ds, generator_model):
    i = 1
    for img in input_ds:
        prediction = generator_model(img, training=False)[0].numpy()  # make predition
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)  # re-scale
        im = PIL.Image.fromarray(prediction)
        im.save(os.path.join(path, "images", '{}.jpg'.format(i)))
        i += 1

        # # TODO: Delete me
        # if i == 30:
        #     break


class LogCallback(Callback):
    def __init__(self, logger, log_interval):
        self.logger = logger
        self.log_interval = log_interval

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.log_interval == 0:
            self.logger.info(
                "| \t\tbatch {} | monet_gen_loss: {:2.5f}, photo_gen_loss: {:2.5f}, monet_disc_loss: {:2.5f},"
                "photo_disc_loss: {:2.5f}, total_cycle_loss: {:2.5f}".format(batch, logs["monet_gen_loss"],
                    logs["photo_gen_loss"], logs["monet_disc_loss"], logs["photo_disc_loss"], logs["total_cycle_loss"])
            )

    # def on_test_batch_end(self, batch, logs=None):
    #     print(
    #         "Up to batch {}, the average loss is {:7.2f}.".format(batch, logs["loss"])
    #     )

    def on_epoch_end(self, epoch, logs=None):
        self.logger.info(
            "| End of Epoch {} | monet_gen_loss: {:2.5f}, photo_gen_loss: {:2.5f}, monet_disc_loss: {:2.5f},"
            "photo_disc_loss: {:2.5f}, total_cycle_loss: {:2.5f}".format(epoch + 1, logs["monet_gen_loss"],
                                                                         logs["photo_gen_loss"],
                                                                         logs["monet_disc_loss"],
                                                                         logs["photo_disc_loss"],
                                                                         logs["total_cycle_loss"])
        )


class ClearMemory(Callback):
    def __init__(self, logger):
        self.logger = logger

    def on_epoch_end(self, epoch, logs=None):
        backend.clear_session()
        gc.collect()
        tf.compat.v1.reset_default_graph()
        self.logger.info(" | Resource Report: {}".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
