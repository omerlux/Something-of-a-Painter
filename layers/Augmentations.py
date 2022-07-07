import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import Model

# Augmentations functions
def rand_brightness(x):
    factor = 1  # 1    # between (-factor/2, factor/2) around 0
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) * factor - factor / 2          # - 0.5
    x = x + magnitude
    return x


def rand_saturation(x):
    factor = 2  # 2    # between (1 - factor/2, 1 + factor/2) - around 1
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) * factor + (1 - factor / 2)    # * 2
    x_mean = tf.reduce_sum(x, axis=3, keepdims=True) * 0.3333333333333333333
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_contrast(x):
    magnitude = tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) + 0.5
    x_mean = tf.reduce_sum(x, axis=[1, 2, 3], keepdims=True) * 5.086e-6
    x = (x - x_mean) * magnitude + x_mean
    return x


def rand_translation(x, ratio=0.125):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    shift = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    translation_x = tf.random.uniform([batch_size, 1], -shift[0], shift[0] + 1, dtype=tf.int32)
    translation_y = tf.random.uniform([batch_size, 1], -shift[1], shift[1] + 1, dtype=tf.int32)
    grid_x = tf.clip_by_value(tf.expand_dims(tf.range(image_size[0], dtype=tf.int32), 0) + translation_x + 1, 0,
                              image_size[0] + 1)
    grid_y = tf.clip_by_value(tf.expand_dims(tf.range(image_size[1], dtype=tf.int32), 0) + translation_y + 1, 0,
                              image_size[1] + 1)
    x = tf.gather_nd(tf.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0]]), tf.expand_dims(grid_x, -1), batch_dims=1)
    x = tf.transpose(tf.gather_nd(tf.pad(tf.transpose(x, [0, 2, 1, 3]), [[0, 0], [1, 1], [0, 0], [0, 0]]),
                                  tf.expand_dims(grid_y, -1), batch_dims=1), [0, 2, 1, 3])
    return x


def rand_cutout(x, ratio=0.5):
    batch_size = tf.shape(x)[0]
    image_size = tf.shape(x)[1:3]
    cutout_size = tf.cast(tf.cast(image_size, tf.float32) * ratio + 0.5, tf.int32)
    offset_x = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[0] + (1 - cutout_size[0] % 2),
                                 dtype=tf.int32)
    offset_y = tf.random.uniform([tf.shape(x)[0], 1, 1], maxval=image_size[1] + (1 - cutout_size[1] % 2),
                                 dtype=tf.int32)
    grid_batch, grid_x, grid_y = tf.meshgrid(tf.range(batch_size, dtype=tf.int32),
                                             tf.range(cutout_size[0], dtype=tf.int32),
                                             tf.range(cutout_size[1], dtype=tf.int32), indexing='ij')
    cutout_grid = tf.stack(
        [grid_batch, grid_x + offset_x - cutout_size[0] // 2, grid_y + offset_y - cutout_size[1] // 2], axis=-1)
    mask_shape = tf.stack([batch_size, image_size[0], image_size[1]])
    cutout_grid = tf.maximum(cutout_grid, 0)
    cutout_grid = tf.minimum(cutout_grid, tf.reshape(mask_shape - 1, [1, 1, 1, 3]))
    mask = tf.maximum(
        1 - tf.scatter_nd(cutout_grid, tf.ones([batch_size, cutout_size[0], cutout_size[1]], dtype=tf.float32),
                          mask_shape), 0)
    x = x * tf.expand_dims(mask, axis=3)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}


def DiffAugment(x, policy=["color", "translation", "cutout"], channels_first=False):
    if policy:
        if channels_first:
            x = tf.transpose(x, [0, 2, 3, 1])
        for p in policy:
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if channels_first:
            x = tf.transpose(x, [0, 3, 1, 2])
    return x    # tf.clip_by_value(x, clip_value_min=-1, clip_value_max=1)      # applying clipping to [-1, 1] values


def DataAugment(height=256, width=256, channels=3):
    inputs = L.Input(shape=[height, width, channels])

    flip = tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal')
    rotate = tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-.25, 0.25), fill_mode='reflect')
    crop = tf.keras.layers.experimental.preprocessing.RandomCrop(int(height / 1.5), int(width / 1.5))
    resize = tf.keras.layers.experimental.preprocessing.Resizing(height, width)

    outputs = flip(inputs)
    outputs = rotate(outputs)
    outputs = crop(outputs)
    outputs = resize(outputs)
    return Model(inputs=inputs, outputs=outputs)
