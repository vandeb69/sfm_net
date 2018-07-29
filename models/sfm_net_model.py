from base.base_model import BaseModel
import tensorflow as tf

from modules.sfm_net.structure import structure_net
from modules.sfm_net.motion import motion_net
from modules.sfm_net.loss import get_frame_loss, get_smooth_loss, get_fb_depth_loss


class SfmNetModel(BaseModel):
    def __init__(self, config):
        super(SfmNetModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        img_h = self.config.image_height
        img_w = self.config.image_width
        img_c = self.config.num_channels

        with tf.variable_scope('inputs'):
            self.is_training = tf.placeholder(tf.bool)

            self.I_t0 = tf.placeholder(tf.float32, {None, img_h, img_w, img_c}, name='frame_t0')
            self.I_t1 = tf.placeholder(tf.float32, {None, img_h, img_w, img_c}, name='frame_t1')

        self.f_point_cloud_1, self.f_depth_output = structure_net(self.I_t0)
        self.b_point_cloud_1, self.b_depth_output = structure_net(self.I_t1, reuse=tf.AUTO_REUSE)

        self.f_pix_pos, self.f_flow, self.f_point_cloud_2, self.f_motion_map = motion_net(self.I_t0, self.I_t1,
                                                                                          self.f_point_cloud_1)
        self.b_fix_pos, self.b_flow, self.b_point_cloud_2, self.b_motion_map = motion_net(self.I_t1, self.I_t1,
                                                                                          self.b_point_cloud_1,
                                                                                          reuse=tf.AUTO_REUSE)

        with tf.name_scope('loss'):
            self.f_frame_loss = get_frame_loss()(self.I_t0, self.I_t1, self.f_pix_pos)
            self.b_frame_loss = get_frame_loss()(self.I_t1, self.I_t0, self.b_fix_pos, reuse=tf.AUTO_REUSE)

            self.f_flow_sm_loss = get_smooth_loss(order=1)(self.f_flow, 'flow')
            self.b_flow_sm_loss = get_smooth_loss(order=1)(self.b_flow, 'flow', reuse=tf.AUTO_REUSE)

            self.f_depth_sm_loss = get_smooth_loss(order=1)(self.f_depth_output, 'depth')
            self.b_depth_sm_loss = get_smooth_loss(order=1)(self.b_depth_output, 'depth', reuse=tf.AUTO_REUSE)

            self.f_motion_sm_loss = get_smooth_loss(order=1)(self.f_motion_map, 'motion')
            self.b_motion_sm_loss = get_smooth_loss(order=1)(self.b_motion_map, 'motion', reuse=tf.AUTO_REUSE)

            self.f_depth_loss = get_fb_depth_loss()(self.f_depth_output, self.b_depth_output, self.f_pix_pos,
                                                    self.f_point_cloud_1)
            self.b_depth_loss = get_fb_depth_loss()(self.b_depth_output, self.f_depth_output, self.b_pix_pos,
                                                    self.b_point_cloud_1, reuse=tf.AUTO_REUSE)

            self.total_loss = self.f_depth_loss + self.f_depth_sm_loss + self.f_motion_sm_loss + self.f_flow_sm_loss + \
                              self.f_frame_loss + self.b_depth_loss + self.b_depth_sm_loss + self.b_motion_sm_loss + \
                              self.b_flow_sm_loss + self.b_frame_loss

            self.train_step = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate,
                                                     beta1=self.config.beta1).minimize(self.total_loss)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


