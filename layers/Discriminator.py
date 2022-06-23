from layers.UpDownSample import downsample, upsample
from layers.EncDecTrans import encoder_block, transformer_block, decoder_block

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
import tensorflow.keras.layers as L


def discriminator_fn(hieght, width, channels):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = L.Input(shape=[hieght, width, channels], name='input_image')

    x = inp

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

    zero_pad1 = L.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = L.Conv2D(512, 4, strides=1,
                    kernel_initializer=initializer,
                    use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(conv)

    leaky_relu = L.LeakyReLU()(norm1)

    zero_pad2 = L.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

    last = L.Conv2D(1, 4, strides=1,
                    kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

    return Model(inputs=inp, outputs=last)


def res_discriminator_fn(height, width, channels):
    conv_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.02)

    inputs = L.Input(shape=[height, width, channels], name='input_image')
    # inputs_patch = L.experimental.preprocessing.RandomCrop(height=70, width=70, name='input_image_patch')(inputs) # Works only with GPU

    # Encoder
    x = encoder_block(inputs, 64, 4, 2, apply_instancenorm=False, activation=L.LeakyReLU(0.2),
                      name='block_1')  # (bs, 128, 128, 64)
    x = encoder_block(x, 128, 4, 2, apply_instancenorm=True, activation=L.LeakyReLU(0.2),
                      name='block_2')  # (bs, 64, 64, 128)
    x = encoder_block(x, 256, 4, 2, apply_instancenorm=True, activation=L.LeakyReLU(0.2),
                      name='block_3')  # (bs, 32, 32, 256)
    x = encoder_block(x, 512, 4, 1, apply_instancenorm=True, activation=L.LeakyReLU(0.2),
                      name='block_4')  # (bs, 32, 32, 512)

    outputs = L.Conv2D(1, 4, strides=1, padding='valid', kernel_initializer=conv_initializer)(x)  # (bs, 29, 29, 1)

    discriminator = Model(inputs, outputs)

    return discriminator
