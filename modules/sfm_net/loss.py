import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D, Permute


class get_frame_loss():
    def __init__(self):
        self.output_size = [128, 384]

    def __call__(self, frame0, frame1, pos_2d_new, reuse=False):  # shape [b, h, w, 3] shape [b, h, w, 3], shape [b, 2, w * h]
        with tf.variable_scope('frame_loss', reuse=reuse):
            batch_size = tf.shape(frame1)[0]
            height = 128
            width = 384
            num_channels = 3
            output_height = 128
            output_width = 384

            x_s = tf.slice(pos_2d_new, [0, 0, 0], [-1, 1, -1])  # shape [b, 1, w * h]
            y_s = tf.slice(pos_2d_new, [0, 1, 0], [-1, 1, -1])  # shape [b, 1, w * h]
            x_s_flatten = tf.reshape(x_s, [-1])  # shape [b * w * h]
            y_s_flatten = tf.reshape(y_s, [-1])  # shape [b * w * h]

            transformed_image = self._interpolate(frame1, x_s_flatten, y_s_flatten, self.output_size)

            transformed_image = tf.reshape(transformed_image, shape=(-1, output_height, output_width, num_channels))

            loss = self.compute_loss(frame0, transformed_image)

            return loss

    def compute_loss(self, frame0, transformed_image):
        loss = tf.reduce_mean(tf.abs(tf.subtract(frame0, transformed_image)))
        return loss

    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = 128
        width = 384
        num_channels = tf.shape(image)[3]

        x = tf.cast(x, dtype='float32')  # shape [b * w * h]
        y = tf.cast(y, dtype='float32')  # shape [b * w * h]

        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width = output_size[1]

        x = x*(width_float)  # shape [b * w * h]
        y = y*(height_float)  # shape [b * w * h]

        x0 = tf.cast(tf.floor(x), 'int32')  # shape [b * w * h]
        x1 = x0 + 1  # shape [b * w * h]
        y0 = tf.cast(tf.floor(y), 'int32')  # shape [b * w * h]
        y1 = y0 + 1  # shape [b * w * h]

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1, dtype='int32')

        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)  # shape [b * w * h]
        x1 = tf.clip_by_value(x1, zero, max_x)  # shape [b * w * h]
        y0 = tf.clip_by_value(y0, zero, max_y)  # shape [b * w * h]
        y1 = tf.clip_by_value(y1, zero, max_y)  # shape [b * w * h]

        flat_image_dimensions = width * height

        pixels_batch = tf.range(batch_size)*flat_image_dimensions  # shape [b]  [0, w * h, 2 * w * h, ..., b * w * h]
        flat_output_dimensions = output_height * output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)  # shape [b * w * h]
        base_y0 = base + y0 * width  # shape [b * w * h]
        base_y1 = base + y1 * width  # shape [b * w * h]
        indices_a = base_y0 + x0  # shape [b * w * h]
        indices_b = base_y1 + x0  # shape [b * w * h]
        indices_c = base_y0 + x1  # shape [b * w * h]
        indices_d = base_y1 + x1  # shape [b * w * h]

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a * pixel_values_a,
                           area_b * pixel_values_b,
                           area_c * pixel_values_c,
                           area_d * pixel_values_d])
        return output

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')  # shape [1, w * h]
        x = tf.reshape(x, shape=(-1, 1))  # shape [b, 1]
        x = tf.matmul(x, ones)  # shape [b, w * h]
        return tf.reshape(x, [-1])  # shape [b * w * h]


class get_smooth_loss():
    def __init__(self, kernel=[[1, 2, 1], [0, 0, 0], [-1, -2, -1]], order=1):
        self.kernel = np.array(kernel)
        self.order = order

    def build(self, field_c):
        v_kernel = self.kernel
        h_kernel = self.kernel.T
        h_init = keras.initializers.Constant(value=h_kernel)
        v_init = keras.initializers.Constant(value=v_kernel)

        self.conv_h = Conv2D(filters=field_c, kernel_size=3, strides=1, kernel_initializer=h_init, padding='same')
        self.conv_h.trainable = False

        self.conv_v = Conv2D(filters=field_c, kernel_size=3, strides=1, kernel_initializer=v_init, padding='same')
        self.conv_v.trainable = False

    def compute_gradient(self, field):
        loss_v = self.conv_v(field)
        loss_h = self.conv_h(field)

        gradient_loss = loss_h + loss_v

        return gradient_loss

    def compute_loss(self, field):
        f1_gradient_loss = self.compute_gradient(field)

        if self.order == 1:
            loss = tf.reduce_mean(tf.abs(f1_gradient_loss), -1)
            loss = tf.reduce_mean(loss)

        if self.order == 2:
            f2_gradient_loss = self.compute_gradient(f1_gradient_loss)

            loss = tf.reduce_mean(tf.abs(f2_gradient_loss), -1)
            loss = tf.reduce_mean(loss)

        return loss

    def __call__(self, field, loss_type=None, reuse=False):
        with tf.variable_scope(loss_type, reuse=reuse):
            if loss_type == 'flow':
                field = Permute((2, 1))(field)
                field = tf.reshape(field, (-1, 128, 384, 2))

            field_c = field.shape.as_list()[-1]

            self.build(field_c)
            loss = self.compute_loss(field)

            return loss


class get_fb_depth_loss():
    def __init__(self):
        self.output_size=[128, 384]

    def __call__(self, depth0, depth1, pos_2d_new, motion, reuse=False):
        with tf.variable_scope('fb_depth_loss', reuse=reuse):
            batch_size = tf.shape(depth0)[0]
            height = tf.shape(depth0)[1]
            width = tf.shape(depth0)[2]
            num_channels = tf.shape(depth0)[3]
            output_height = self.output_size[0]
            output_width = self.output_size[1]

            x_s = tf.slice(pos_2d_new, [0, 0, 0], [-1, 1, -1], name='err')
            y_s = tf.slice(pos_2d_new, [0, 1, 0], [-1, 1, -1])
            x_s_flatten = tf.reshape(x_s, [-1])
            y_s_flatten = tf.reshape(y_s, [-1])

            transformed_depth1 = self._interpolate(depth1, x_s_flatten, y_s_flatten, self.output_size)
            transformed_depth1 = tf.reshape(transformed_depth1, shape=(-1, output_height, output_width, num_channels))

            motion_z = tf.slice(motion, [0, 2, 0], [-1, 1, -1])
            motion_z = tf.reshape(motion_z, (-1, output_height, output_width, 1))
            transformed_depth0 = tf.add(depth0, motion_z)
            loss = self.compute_loss(transformed_depth0, transformed_depth1)

        return loss

    def compute_loss(self, transformed_depth0, transformed_depth1):
        loss = tf.reduce_mean(tf.abs(tf.subtract(transformed_depth0, transformed_depth1)))
        return loss

    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        num_channels = tf.shape(image)[3]

        x = tf.cast(x, dtype='float32')
        y = tf.cast(y, dtype='float32')

        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width = output_size[1]

        x = x * (width_float)
        y = y * (height_float)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1, dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width * height
        pixels_batch = tf.range(batch_size) * flat_image_dimensions
        flat_output_dimensions = output_height * output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a * pixel_values_a,
                           area_b * pixel_values_b,
                           area_c * pixel_values_c,
                           area_d * pixel_values_d])
        return output

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1, 1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])





