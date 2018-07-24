import tensorflow as tf

class get_frame_loss():
    def __init__(self):
        self.output_size = [128, 384]

    def __call__(self, frame0, frame1, pos_2d_new, reuse=False):
        with tf.variable_scope('frame_loss', reuse=reuse):
            batch_size = tf.shape(frame1)[0]
            height = 128
            width = 384
            num_channels = 3
            output_height = 128
            output_width = 384

            x_s = tf.slice(pos_2d_new, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(pos_2d_new, [0, 1, 0], [-1, 1, -1])
            x_s_flatten = tf.reshape(x_s, [-1])
            y_s_flatten = tf.reshape(y_s, [-1])

            transformed_image = self._interpolate(frame1, x_s_flatten, y_s_flatten, self.output_size)

            transformed_image = tf.reshape(transformed_image, shape=(-1, output_height, output_width, num_channels))

            loss = self.compute_loss(frame0, transformed_image)

            return loss

    def compute_loss(self, frame0, transformed_image):
        pass