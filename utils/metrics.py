from tensorflow.keras import losses
import tensorflow as tf
import numpy as np


# Discriminator loss:
def discriminator_loss(real, generated):
    real_loss = losses.BinaryCrossentropy(from_logits=True, reduction=losses.Reduction.AUTO)(tf.ones_like(real), real)

    generated_loss = losses.BinaryCrossentropy(from_logits=True, reduction=losses.Reduction.AUTO)(
        tf.zeros_like(generated), generated)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss * 0.5


def discriminator_loss_hinge(real, generated):
    real_loss = tf.math.minimum(tf.zeros_like(real), real - tf.ones_like(real))

    generated_loss = tf.math.minimum(tf.zeros_like(generated), -generated - tf.ones_like(generated))

    total_disc_loss = real_loss + generated_loss

    return tf.reduce_mean(-total_disc_loss * 0.5)


# Generator loss
def generator_loss(generated):
    return losses.BinaryCrossentropy(from_logits=True, reduction=losses.Reduction.AUTO)(tf.ones_like(generated),
                                                                                        generated)

def generator_loss_minus(generated):
    return losses.BinaryCrossentropy(from_logits=True, reduction=losses.Reduction.AUTO)(tf.ones_like(generated),
                                                                                        generated)


# Cycle consistency loss (measures if original photo and the twice transformed photo to be similar to one another)
def calc_cycle_loss(real_image, cycled_image, LAMBDA):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

    return LAMBDA * loss1


# Identity loss (compares the image with its generator (i.e. photo with photo generator))
def identity_loss(real_image, same_image, LAMBDA):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss


def load_inception_model():
    """
    :return: pretrained inception_model for FID evaluation
    """
    inception_model = tf.keras.applications.InceptionV3(input_shape=(256, 256, 3), pooling="avg", include_top=False)

    mix3 = inception_model.get_layer("mixed9").output
    f0 = tf.keras.layers.GlobalAveragePooling2D()(mix3)

    inception_model = tf.keras.Model(inputs=inception_model.input, outputs=f0)
    inception_model.trainable = False
    return inception_model


# --------------------- MiFID Functions ---------------------


def calculate_activation_statistics_mod(images, fid_model):
    """

    :param images: input images to calculate statistics on
    :param fid_model: the model which the statisits are created from
    :return: mu, sigma
    """
    act = tf.cast(fid_model.predict(images), tf.float32)

    mu = tf.reduce_mean(act, axis=0)
    mean_x = tf.reduce_mean(act, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(act), act) / tf.cast(tf.shape(act)[0], tf.float32)
    sigma = vx - mx
    return mu, sigma, act


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """
    :param mu1:
    :param sigma1:
    :param mu2:
    :param sigma2:
    :return: calculate the frechet distance between 2 Gaussian distributions
    """
    fid_epsilon = 1e-14

    covmean = tf.linalg.sqrtm(tf.cast(tf.matmul(sigma1, sigma2), tf.complex64))
    #         isgood=tf.cast(tf.math.is_finite(covmean), tf.int32)
    #         if tf.size(isgood)!=tf.math.reduce_sum(isgood):
    #             return 0

    covmean = tf.cast(tf.math.real(covmean), tf.float32)

    tr_covmean = tf.linalg.trace(covmean)

    return tf.matmul(tf.expand_dims(mu1 - mu2, axis=0), tf.expand_dims(mu1 - mu2, axis=1)) + tf.linalg.trace(
        sigma1) + tf.linalg.trace(sigma2) - 2 * tr_covmean


def normalize_rows(x: np.ndarray):
    """
    function that normalizes each row of the matrix x to have unit length.

    Args:
     ``x``: A numpy matrix of shape (n, m)

    Returns:
     ``x``: The normalized (by row) numpy matrix.
    """
    return np.nan_to_num(x/np.linalg.norm(x, ord=2, axis=1, keepdims=True))


def cosine_distance(features1, features2):
    features1_nozero = features1[np.sum(features1, axis=1) != 0]
    features2_nozero = features2[np.sum(features2, axis=1) != 0]
    norm_f1 = normalize_rows(features1_nozero)
    norm_f2 = normalize_rows(features2_nozero)

    d = 1.0 - np.abs(np.matmul(norm_f1, norm_f2.T))
    # print('d.shape=',d.shape)
    # print('np.min(d, axis=1).shape=',np.min(d, axis=1).shape)
    mean_min_d = np.mean(np.min(d, axis=1))
    # print('distance=',mean_min_d)
    return mean_min_d


def distance_thresholding(d, eps=1e-14):
    if d < eps:
        return d
    else:
        return 1


def MiFID(images, gen_model, inception_model, mu2, sigma2, features2):
    """

    :param images:
    :param gen_model:
    :param inception_model:
    :param mu2: statistics of the known dataset
    :param sigma2: statistics of the known dataset
    :param features2:
    :return: MiFID score for the images that created from the gen_model, in compare to the known mu2,sigma2
    """
    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    x = gen_model(inp)
    x = inception_model(x)
    fid_model = tf.keras.Model(inputs=inp, outputs=x)
    eps = 1e-14

    # calculating FID:
    mu1, sigma1, features1 = calculate_activation_statistics_mod(images, fid_model)
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    # calculating Cosine similarity:
    distance = cosine_distance(features1, features2)
    distance = distance_thresholding(distance, eps)

    return float(fid_value / (distance + eps))
