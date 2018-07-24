import tensorflow as tf
import keras
from keras.layers import Conv2D, Activation
from convDeconv import conv_deconv_net


def clip_relu(x):
    x = tf.clip_by_value(x, 1, 100)
    return x


def depth_net(frame):
    top, _ = conv_deconv_net(frame)  # shape [1, w, h, 32]
    top = Conv2D(filters=1, kernel_size=1, strides=1, padding='same', kernel_initializer=keras.initializers.glorot_normal())(top)  # shape [1, w, h, 1]
    depth = Activation(clip_relu)(top)  # shape [1, w, h, 1]
    return depth


class CloudTransformer():
    def __init__(self, intrinsics=(0.5, 0.5, 1.0), **kwargs):
        self.output_dim = 3
        self.cam_intrinsics = intrinsics
        self.build()

    def build(self):
        self.cx_ = self.cam_intrinsics[0]
        self.cy_ = self.cam_intrinsics[1]
        self.cf_ = self.cam_intrinsics[2]

        self.cx = tf.constant(self.cam_intrinsics[0], dtype=tf.float32)
        self.cy = tf.constant(self.cam_intrinsics[1], dtype=tf.float32)
        self.cf = tf.constant(self.cam_intrinsics[2], dtype=tf.float32)

    def mesh_grid(self, width, height):
        """
        [(xi / w - cx) / f, (yi / h - cy) / f, 1]

        next just

        d * [(xi / w - cx) / f, (yi / h - cy) / f, 1]
        to get [Xi, Yi, Zi]
        """

        x_linspace = tf.linspace(-self.cx_, 1 - self.cx_, width)  # shape [w]
        y_linspace = tf.linspace(-self.cy_, 1 - self.cy_, height)  # shape [h]

        y_cord, x_cord = tf.meshgrid(y_linspace, x_linspace)  # shapes [w, h]

        x_cord = tf.reshape(x_cord, [-1])  # shape [w * h]
        y_cord = tf.reshape(y_cord, [-1])  # shape [w * h]

        f_ = tf.ones_like(x_cord)  # shape [w * h]

        x_ = tf.div(x_cord, self.cf)  # shape [w * h]
        y_ = tf.div(y_cord, self.cf)  # shape [w * h]

        grid = tf.concat([x_, y_, f_], 0)  # shape [3 * w * h]
        return grid

    def transform(self, x):
        # get input shape
        batch_size = tf.shape(x)[0]
        width = tf.shape(x)[1]
        height = tf.shape(x)[2]
        channel = tf.shape(x)[3]
        batch_size = tf.cast(batch_size, tf.int32)
        width = tf.cast(width, tf.int32)
        height = tf.cast(height, tf.int32)
        channel = tf.cast(channel, tf.int32)

        # grid
        grid = self.mesh_grid(width, height)  # shape [3 * w * h]
        grid = tf.expand_dims(grid, 0)  # shape [1, 3 * w * h]
        grid = tf.reshape(grid, [-1])  # shape [3 * w * h]

        grid_stack = tf.tile(grid, tf.stack([batch_size]))  # shape [b * 3 * w * h]
        grid_stack = tf.reshape(grid_stack, [batch_size, 3, -1])  # shape [b, 3, w * h]
        depth = tf.reshape(x, [batch_size, 1, -1])  # shape [b, 1, w * h]
        depth = tf.concat([depth] * self.output_dim, 1)  # shape [b, 3, w * h]

        point_cloud = tf.multiply(depth, grid_stack)  # shape [b, 3, w * h]

        return point_cloud

    def __call__(self, x):
        point_cloud = self.transform(x)
        return point_cloud


def structure_net(input_frame, reuse=False):
    with tf.variable_scope('structure_net', reuse=reuse):
        depth_output = depth_net(input_frame)  # shape [b, w, h, 1]
        point_cloud_output = CloudTransformer()(depth_output)  # shape [b, 3, w * h]
        return point_cloud_output, depth_output
