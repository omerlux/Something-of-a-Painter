import datetime
import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import PIL
from layers.Augmentations import DiffAugment

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
        os.mkdir(os.path.join(path, 'generated'))
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
    path = os.path.join(path, "images", f"{name}.png")
    fig.suptitle(f"{name}")
    plt.savefig(path)
    plt.show()


def display_augmented_samples(path, name, ds, num_images=1):
    ds_iter = iter(ds)

    fig = plt.figure(figsize=(5 * num_images, 10))
    for j in range(num_images):
        example_sample = next(ds_iter)
        img = example_sample[0] * 0.5 + 0.5
        x = DiffAugment(img[None, :, :, :])[0]
        ax = plt.subplot(2, num_images, j + 1)
        plt.axis('off')
        ax.set_title("origin")
        ax.imshow(img)
        ax = plt.subplot(2, num_images, (j + num_images) + 1)
        plt.axis('off')
        ax.set_title("augmented")
        ax.imshow(x)
    path = os.path.join(path, "images", f"{name}_augmented.png")
    fig.suptitle(f"{name} augmented")
    plt.savefig(path)
    plt.show()


def display_generated_samples(path, ds, model, n_samples):
    ds_iter = iter(ds)
    for n_sample in range(n_samples):
        example_sample = next(ds_iter)
        generated_sample = model.predict(example_sample)

        plt.subplot(121)
        plt.title("input image")
        plt.imshow(example_sample[0] * 0.5 + 0.5)
        plt.axis('off')

        plt.subplot(122)
        plt.title("Generated image")
        plt.imshow(generated_sample[0] * 0.5 + 0.5)
        plt.axis('off')

        path_tmp = os.path.join(path, "images", "{0:02d}.png".format(n_sample))
        plt.savefig(path_tmp)

        plt.show()


def predict_and_save(path, input_ds, generator_model):
    i = 1
    for img in input_ds:
        prediction = generator_model(img, training=False)[0].numpy()  # make predition
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)  # re-scale
        im = PIL.Image.fromarray(prediction)
        im.save(os.path.join(path, "generated", '{}.jpg'.format(i)))
        i += 1
