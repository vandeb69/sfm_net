import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D, Dense, Reshape, Activation
from modules.sfm_net.utils import conv_deconv_net


def sin_relu(x):
    x = tf.clip_by_value(x, -1., 1.)
    return x


def param_net(frame_t0, frame_t1, k_obj=4, ):
    init = keras.initializers.TruncatedNormal(mean=0.0, stddev=0.0001)
    frame_pair = tf.concat([frame_t0, frame_t1], -1)  # shape [b, w, h, 2 * c]
    top, embed = conv_deconv_net(frame_pair)  # shape [b, w, h, 32], shape [b, w/32, h/32, 1024]
    mask = Conv2D(filters=k_obj, kernel_size=1, strides=1, padding='same', kernel_initializer=init)(top)  # shape [b, w, h, k_obj]

    embed = Dense(512, kernel_initializer=init)(embed)  # shape [b, 1, 1, 512]
    embed = Dense(512, kernel_initializer=init)(embed)  # shape [b, 1, 1, 512]
    embed = Reshape([-1])(embed)  # shape [b, 512]

    cam_t_ = Dense(3, kernel_initializer=init)(embed)  # shape [b, 3]
    cam_t = Activation('relu')(cam_t_)  # shape [b, 3]

    cam_p = Dense(600, kernel_initializer=init)(embed)  # shape [b, 600]
    cam_p = Activation('relu')(cam_p)  # shape [b, 600]

    cam_r = Dense(3, kernel_initializer=init)(embed)  # shape [b, 3]
    cam_r = Activation(sin_relu)(cam_r)  # shape [b, 3]

    obj_mask = Activation('sigmoid')(mask)  # shape [b, w, h, k_obj]

    obj_t = Activation('relu')(Dense(3 * k_obj, kernel_initializer=init)(embed))  # shape [b, 3 * k_obj]
    obj_t = tf.reshape(obj_t, (-1, k_obj, 3))  # shape [b, k_obj, 3]

    obj_p = Activation('relu')(Dense(600 * k_obj, kernel_initializer=init)(embed))  # shape [b, 600 * k_obj]
    obj_p = tf.reshape(obj_p, (-1, k_obj, 600))  # shape [b, k_obj, 600]

    obj_r = Activation(sin_relu)(Dense(3 * k_obj, kernel_initializer=init)(embed))  # shape [b, 3 * k_obj]
    obj_r = tf.reshape(obj_r, (-1, k_obj, 3))  # shape [b, k_obj, 3]

    return [cam_t, cam_p, cam_r], [obj_t, obj_p, obj_r, obj_mask]


class optical_transformer():

    def __init__(self, intrinsics=(0.5, 0.5, 1.0), img_shape=(384, 128), **kwargs):
        self.cam_intrinsics = intrinsics
        self.img_w = self.np_tf(img_shape[0])
        self.img_h = self.np_tf(img_shape[1])
        self.img_w_ = int(img_shape[0])
        self.img_h_ = int(img_shape[1])

        self.cx_ = self.cam_intrinsics[0]
        self.cy_ = self.cam_intrinsics[1]
        self.cf_ = self.cam_intrinsics[2]

        self.cx = self.np_tf(self.cam_intrinsics[0])
        self.cy = self.np_tf(self.cam_intrinsics[1])
        self.cf = self.np_tf(self.cam_intrinsics[2])

        so3_a = np.array([                      # shape [3, 9]
            [0, -1, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        so3_b = np.array([                      # shape [3, 9]
            [0, 0, 1, 0, 0, 0, -1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0]
        ])

        so3_y = np.array([                      # shape [3, 9]
            [0, 0, 0, 0, 0, -1, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

        self.so3_a = self.np_tf(so3_a)
        self.so3_b = self.np_tf(so3_b)
        self.so3_y = self.np_tf(so3_y)

    def np_tf(self, array):
        return tf.constant(array, tf.float32)

    def build(self, cam_motion, obj_motion, x):
        self.cam_motion = cam_motion  # [shape [b, 3], shape [b, 600], shape [b, 3]]
        self.obj_motion = obj_motion  # [shape [b, k_obj, 3], shape [b, k_obj, 600], shape [b, k_obj, 3], shape [b, w, h, k_obj]]
        self.mask_size = obj_motion[0].shape.as_list()[1]  # k_obj
        self.x_shape = x.shape.as_list()  # [b, 3, w * h * c]

    def so3_mat(self, sin):
        sin = tf.expand_dims(sin, -1)  # shape [b, 3, 1]
        cos = tf.sqrt(tf.ones_like(sin) - tf.square(sin))  # shape [b, 3, 1]

        t = tf.concat([sin, cos, tf.ones_like(sin)], -1)  # shape [b, 3, 3]
        t_a = tf.slice(t, [0, 0, 0], [-1, 1, -1])  # shape [b, 1, 3] (=t[:, 0, :])
        t_b = tf.slice(t, [0, 1, 0], [-1, 1, -1])  # shape [b, 1, 3] (=t[:, 1, :])
        t_y = tf.slice(t, [0, 2, 0], [-1, 1, -1])  # shape [b, 1, 3] (=t[:, 2, :])

        t_a = tf.reshape(t_a, (-1, 3))  # shape [b, 3]
        t_b = tf.reshape(t_b, (-1, 3))  # shape [b, 3]
        t_y = tf.reshape(t_y, (-1, 3))  # shape [b, 3]

        soa = tf.matmul(t_a, self.so3_a)  # shape [b, 9]
        soa = tf.reshape(soa, (-1, 3, 3))  # shape [b, 3, 3]

        sob = tf.matmul(t_b, self.so3_b)  # shape [b, 9]
        sob = tf.reshape(sob, (-1, 3, 3))  # shape [b, 3, 3]

        soy = tf.matmul(t_y, self.so3_y)  # shape [b, 9]
        soy = tf.reshape(soy, (-1, 3, 3))  # shape [b, 3, 3]

        so3 = tf.matmul(soa, tf.matmul(sob, soy))  # shape [b, 3, 3]
        return so3

    def pior_pont(self, p):
        batch_size = p.shape.as_list()[0]  # b
        p_ret = tf.reshape(p, (-1, 30, 20))  # shape [b, 30, 20] or shape [b * k_obj, 30, 20]
        p_y = tf.reduce_sum(p_ret, 1)  # shape [b, 20] or shape [b * k_obj, 20]
        p_x = tf.reduce_sum(p_ret, 2)  # shape [b, 30] or shape [b * k_obj, 30]
        x_loc = tf.linspace(-30.0, 30.0, 30)
        y_loc = tf.linspace(-20.0, 20.0, 20)
        P_x_loc = tf.reduce_mean(tf.multiply(p_x, x_loc))  # shape [b, 1] or shape [b * k_obj, 1]
        P_x_loc = tf.reshape(P_x_loc, (-1, 1))  # shape [b, 1] or shape [b * k_obj, 1]
        P_y_loc = tf.reduce_mean(tf.multiply(p_y, y_loc))  # shape [b, 1] or shape [b * k_obj, 1]
        P_y_loc = tf.reshape(P_y_loc, (-1, 1))  # shape [b, 1] or shape [b * k_obj, 1]

        ground = tf.ones_like(P_y_loc)  # shape [b, 1] or shape [b * k_obj, 1]
        P = tf.concat([P_x_loc, P_y_loc, ground], 1)  # shape [b, 3] or shape [b * k_obj, 3]

        return P

    def rigid_motion(self, x, R, p, t):
        p = tf.expand_dims(p, -1)  # shape [b, 3, 1] or shape [b * k_obj, 3]
        t = tf.expand_dims(t, -1)   # shape [b, 3, 1] or shape [b * k_obj, 3]
        motion = tf.add(tf.matmul(R, tf.subtract(x, p)), t)  # shape [b, 3, w * h] or shape [b * k_obj, 3, w * h]
        return motion

    def cam_motion_transform(self, x):
        t, p, sin = self.cam_motion  # shape [b, 3], shape [b, 600], shape [b, 3]
        p = self.pior_pont(p)  # shape [b, 3]
        R = self.so3_mat(sin)  # shape [b, 3, 3]
        X = self.rigid_motion(x, R, p, t)  # shape [b, 3, w * h]

        return X

    def obj_motion_transform(self, x_input):
        t, p, sin, mask = self.obj_motion  # shape [b, k_obj, 3], shape [b, k_obj, 600], shape [b, k_obj, 3], shape [b, w, h, k_obj]
        p = self.pior_pont(p)  # shape [b * k_obj, 3]
        sin = tf.reshape(sin, (-1, 3))  # shape [b * k_obj, 3]
        p = tf.reshape(p, (-1, 3))  # shape [b * k_obj, 3]
        t = tf.reshape(t, (-1, 3))  # shape [b * k_obj, 3]

        x_in = tf.expand_dims(x_input, 1)  # shape [b, 1, 3, w * h]
        x_exp = tf.concat([x_in] * self.mask_size, 1)  # shape [b, k_obj, 3, w * h]
        x_ = tf.reshape(x_exp, (-1, 3, 384*128))  # shape [b * k_obj, 3, w * h]

        R = self.so3_mat(sin)  # shape [b, 3, 3]

        x = self.rigid_motion(x_, R, p, t)  # shape [b * k_obj, 3, w * h]

        x = tf.reshape(x, (-1, self.mask_size, 3, 384 * 128))  # shape [b, k_obj, 3, w * h]
        x, motion_map = self.mask_motion(x, mask, x_exp)  # shape [b, 3, w * h], shape [b, k_obj, 3, w * h]
        X = tf.add(x_input, x)  # shape [b, 3, w * h]

        return X, motion_map

    def mask_motion(self, x, mask, x_exp):
        mask = tf.reshape(mask, (-1, self.mask_size, 1, 384*128))  # shape [b, k_obj, 1, w * h]

        x = tf.subtract(x, x_exp)  # shape [b, k_obj, 3, w * h]
        motion_map = tf.multiply(x, mask)  # shape [b, k_obj, 3, w * h]
        x = tf.reduce_sum(motion_map, 1)  # shape [b, 3, w * h]

        return x, motion_map

    def transform_2d(self, x):
        x_3d = tf.slice(x, (0, 0, 0), (-1, 1, 49152))  # shape [b, 1, w * h]
        y_3d = tf.slice(x, (0, 1, 0), (-1, 1, 49152))  # shape [b, 1, w * h]
        z_3d = tf.slice(x, (0, 2, 0), (-1, 1, 49152))  # shape [b, 1, w * h]
        x_z = tf.div(x_3d, z_3d)  # shape [b, 1, w * h]
        y_z = tf.div(y_3d, z_3d)  # shape [b, 1, w * h]

        x_2d = tf.add(tf.multiply(self.cf, x_z), self.cx)  # shape [b, 1, w * h]
        y_2d = tf.add(tf.multiply(self.cf, y_z), self.cy)  # shape [b, 1, w * h]
        pos_2d_new = tf.concat([x_2d, y_2d], 1)  # shape [b, 2, w * h]
        return pos_2d_new

    def get_flow(self, pos_2d_new):
        x_linspace = tf.linspace(0., 1., int(self.img_w_))  # shape [w]
        y_linspace = tf.linspace(0., 1., int(self.img_h_))  # shape [h]
        y_linspace, x_linspace = tf.meshgrid(y_linspace, x_linspace)  # shape [w, h]
        x_linspace = tf.reshape(x_linspace, [1, -1])  # shape [1, w * h]
        y_linspace = tf.reshape(y_linspace, [1, -1])  # shape [1, w * h]
        pos_ori = tf.concat([x_linspace, y_linspace], 0)  # shape [2, w * h]
        flow = tf.subtract(pos_2d_new, pos_ori)  # shape [b, 2, w * h]

        return flow

    def __call__(self, x, cam_motion, obj_motion, ):
        self.build(cam_motion, obj_motion, x)

        point_cloud, motion_map = self.obj_motion_transform(x)  # shape [b, 3, w * h], shape [b, k_obj, 3, w * h]
        point_cloud = self.cam_motion_transform(point_cloud)  # shape [b, 3, w * h]

        pix_pos = self.transform_2d(point_cloud)  # shape [b, 2, w * h]
        flow = self.get_flow(pix_pos)  # shape [b, 2, w * h]

        motion_map = tf.reshape(motion_map, (-1, self.img_h_, self.img_w_, 1))  # shape [b * 3 * k_obj, h, w, 1]
        return pix_pos, flow, point_cloud, motion_map  # shape [b, 2, w * h], shape [b, 2, w * h], shape [b, 3, w * h], shape [b * 3 * k_obj, h, w, 1]


def motion_net(input_frame_0, input_frame_1, point_cloud_0, reuse=False):
    with tf.variable_scope('motion_net', reuse=reuse):
        cam_motion, obj_motion = param_net(input_frame_0, input_frame_1, k_obj=4, )
        pix_pos, flow, point_cloud, motion_map = optical_transformer()(point_cloud_0, cam_motion, obj_motion, )  # shape [b, 2, w * h], shape [b, 2, w * h], shape [b * 3 * k_obj, h, w, 1]
        return pix_pos, flow, point_cloud, motion_map


