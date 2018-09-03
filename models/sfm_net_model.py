from base.base_model import BaseModel
import tensorflow as tf

from modules.sfm_net.structure import structure_net
from modules.sfm_net.motion import motion_net
from modules.sfm_net.loss import get_frame_loss, get_smooth_loss, get_fb_depth_loss

from utils.utils import import_from


class SfmNetModel(BaseModel):
    def __init__(self, config):
        super(SfmNetModel, self).__init__(config)

        Loader = import_from("data_loader.sfm_net", config.data_loader)
        self.data_loader = Loader(config)

        self.build_model()
        self.init_saver()

    def build_model(self):
        img_h = self.config.image_height
        img_w = self.config.image_width
        img_c = self.config.num_channels

        with tf.variable_scope('inputs'):
            self.I_t0, self.I_t1 = self.data_loader.next_batch
            self.is_training = tf.placeholder(tf.bool, name="Training_flag")

            # self.I_t0 = tf.placeholder(tf.float32, (None, img_h, img_w, img_c), name='frame_t0')
            # self.I_t1 = tf.placeholder(tf.float32, (None, img_h, img_w, img_c), name='frame_t1')

        self.f_point_cloud_1, self.f_depth_output = structure_net(self.I_t0, is_training=self.is_training)
        self.b_point_cloud_1, self.b_depth_output = structure_net(self.I_t1, is_training=self.is_training, reuse=tf.AUTO_REUSE)

        self.f_pix_pos, self.f_flow, self.f_point_cloud_2, self.f_motion_map = motion_net(self.I_t0, self.I_t1,
                                                                                          self.f_point_cloud_1,
                                                                                          is_training=self.is_training)
        self.b_pix_pos, self.b_flow, self.b_point_cloud_2, self.b_motion_map = motion_net(self.I_t1, self.I_t0,
                                                                                          self.b_point_cloud_1,
                                                                                          is_training=self.is_training,
                                                                                          reuse=tf.AUTO_REUSE)

        self.f_frame_loss = get_frame_loss()(self.I_t0, self.I_t1, self.f_pix_pos)
        self.b_frame_loss = get_frame_loss()(self.I_t1, self.I_t0, self.b_pix_pos, reuse=tf.AUTO_REUSE)

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

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate,
                                               beta1=self.config.beta1).minimize(self.total_loss)

        print("Model graph was build..")

    def init_data(self, sess):
        self.data_loader.initialize(sess)

        print("Data loader was initialized..")

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

        print("Model saver was initialized..")


if __name__ == "__main__":
    from easydict import EasyDict

    config = EasyDict({"image_height": 128,
                       "image_width": 384,
                       "num_channels": 3,
                       "learning_rate": 0.0003,
                       "beta1": 0.9,
                       "max_to_keep": 5,
                       "batch_size": 1,
                       "data_loader": "SfmNetLoader_DeepTesla_SingleVideo",
                       "video_file": "../data/deeptesla/epochs/epoch01_front.mkv"
                       })
    model = SfmNetModel(config)

    # Loader = import_from("data_loader.sfm_net", config.data_loader)
    # data_loader = Loader(config)

    # run some training iterations as example
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # model.data_loader.initialize(sess)
        sess.run(model.data_loader.iterator.initializer)

        for _ in range(30):
            _, loss = sess.run([model.train_op, model.total_loss], feed_dict={model.is_training: True})
            print(loss)



