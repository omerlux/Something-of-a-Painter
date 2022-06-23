from torch.utils.data import DataLoader
import tensorflow as tf
import os
import numpy as np
import re
import matplotlib.pyplot as plt

global HEIGHT, WIDTH, CHANNELS, AUTO


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=CHANNELS)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [HEIGHT, WIDTH, CHANNELS])
    return image


def read_tfrecord(example):
    tfrecord_format = {
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    return image


def load_dataset(filenames):
    global AUTO
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    return dataset


# def data_augment(image):
#     p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
#     p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
#     #     p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
#
#     # 90º rotations
#     if p_rotate > .8:
#         image = tf.image.rot90(image, k=3)  # rotate 270º
#     elif p_rotate > .6:
#         image = tf.image.rot90(image, k=2)  # rotate 180º
#     elif p_rotate > .4:
#         image = tf.image.rot90(image, k=1)  # rotate 90º
#
#     # Flips
#     image = tf.image.random_flip_left_right(image)
#     image = tf.image.random_flip_up_down(image)
#     if p_spatial > .75:
#         image = tf.image.transpose(image)
#
#     # Train on crops
#     image = tf.image.random_crop(image, size=[HEIGHT, WIDTH, CHANNELS])
#
#     return image


def get_gan_dataset(monet_files, photo_files, repeat=True, shuffle=True, batch_size=1):
    monet_ds = load_dataset(monet_files)
    photo_ds = load_dataset(photo_files)

    if repeat:
        monet_ds = monet_ds.repeat()
        photo_ds = photo_ds.repeat()
    if shuffle:
        monet_ds = monet_ds.shuffle(2048)
        photo_ds = photo_ds.shuffle(2048)

    monet_ds = monet_ds.batch(batch_size, drop_remainder=True)
    photo_ds = photo_ds.batch(batch_size, drop_remainder=True)
    monet_ds = monet_ds.cache()
    photo_ds = photo_ds.cache()
    monet_ds = monet_ds.prefetch(AUTO)
    photo_ds = photo_ds.prefetch(AUTO)

    gan_ds = tf.data.Dataset.zip((monet_ds, photo_ds))

    return gan_ds


def data_provider(args, logger):
    global HEIGHT, WIDTH, CHANNELS, AUTO
    HEIGHT, WIDTH, CHANNELS, AUTO = args.height, args.width, args.channels, args.auto
    IMAGE_SIZE = [HEIGHT, WIDTH]
    GCS_PATH = os.path.join(args.root_path, "gan-getting-started")

    MONET_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/monet_tfrec/*.tfrec'))
    PHOTO_FILENAMES = tf.io.gfile.glob(str(GCS_PATH + '/photo_tfrec/*.tfrec'))

    n_monet_samples = count_data_items(MONET_FILENAMES)
    n_photo_samples = count_data_items(PHOTO_FILENAMES)

    logger.info(f'| Monet TFRecord files: {len(MONET_FILENAMES)}')
    logger.info(f'| Monet image files: {n_monet_samples}')
    logger.info(f'| Photo TFRecord files: {len(PHOTO_FILENAMES)}')
    logger.info(f'| Photo image files: {n_photo_samples}')

    monet_ds = load_dataset(MONET_FILENAMES)
    photo_ds = load_dataset(PHOTO_FILENAMES)
    gan_ds = get_gan_dataset(MONET_FILENAMES, PHOTO_FILENAMES, batch_size=args.batch_size)
    return gan_ds, (n_monet_samples, monet_ds), (n_photo_samples, photo_ds)

