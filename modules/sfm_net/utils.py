import keras
from keras.layers import Conv2D, BatchNormalization, Activation, Concatenate, Deconv2D


def conv(h_0, filters, kernel_size, strides, is_training):
    kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.0001)

    h1 = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                kernel_initializer=kernel_initializer)(h_0)
    h1_bn = BatchNormalization()(h1, training=is_training)
    h1_o = Activation('relu')(h1_bn)
    return h1_o


def deconv(h_0, filters, kernel_size, strides, is_training):
    kernel_initializer = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.0001)

    h1 = Deconv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                  kernel_initializer=kernel_initializer)(h_0)
    h1_bn = BatchNormalization()(h1, training=is_training)
    h1_o = Activation('relu')(h1_bn)
    return h1_o


def conv_deconv_net(frame, is_training):
    h10_o = conv(frame, filters=32, kernel_size=3, strides=1, is_training=is_training)  # shape [b, w, h, 32]
    h11_o = conv(h10_o, filters=64, kernel_size=3, strides=2, is_training=is_training)  # shape [b, w/2, h/2, 64]

    h20_o = conv(h11_o, filters=64, kernel_size=3, strides=1, is_training=is_training)  # shape [b, w/2, h/2, 64]
    h21_o = conv(h20_o, filters=128, kernel_size=3, strides=2, is_training=is_training)  # shape [b, w/4, h/4, 128]

    h30_o = conv(h21_o, filters=128, kernel_size=3, strides=1, is_training=is_training)  # shape [b, w/4, h/4, 128]
    h31_o = conv(h30_o, filters=256, kernel_size=3, strides=2, is_training=is_training)  # shape [b, w/8, h/8, 256]

    h40_o = conv(h31_o, filters=256, kernel_size=3, strides=1, is_training=is_training)  # shape [b, w/8, h/8, 256]
    h41_o = conv(h40_o, filters=512, kernel_size=3, strides=2, is_training=is_training)  # shape [b, w/16, h/16, 512]

    h50_o = conv(h41_o, filters=512, kernel_size=3, strides=1, is_training=is_training)  # shape [b, w/16, h/16, 512]
    h51_o = conv(h50_o, filters=1024, kernel_size=3, strides=2, is_training=is_training)  # shape [b, w/32, h/32, 1024]

    embedding = conv(h51_o, filters=1024, kernel_size=3, strides=1, is_training=is_training)  # shape [b, w/32, h/32, 1024]

    d5 = deconv(embedding, filters=512, kernel_size=3, strides=2, is_training=is_training)  # shape [b, w/16, h/16, 512]
    d4_i = Concatenate(axis=-1)([d5, h50_o])  # shape [b, w/16, h/16, 1024]

    d4 = deconv(d4_i, filters=256, kernel_size=3, strides=2, is_training=is_training)  # shape [b, w/8, h/8, 256]
    d3_i = Concatenate(axis=-1)([d4, h40_o])  # shape [b, w/8, h/8, 512]

    d3 = deconv(d3_i, filters=128, kernel_size=3, strides=2, is_training=is_training)  # shape [b, w/4, h/4, 128]
    d2_i = Concatenate(axis=-1)([d3, h30_o])  # shape [b, w/4, h/4, 256]

    d2 = deconv(d2_i, filters=64, kernel_size=3, strides=2, is_training=is_training)  # shape [b, w/2, h/2, 64]
    d1_i = Concatenate(axis=-1)([d2, h20_o])  # shape [b, w/2, h/2, 128]

    out = deconv(d1_i, filters=32, kernel_size=3, strides=2, is_training=is_training)  # shape [b, w, h, 32]

    return out, embedding  # shape [b, w, h, 32], shape [b, w/32, h/32, 1024]
