from base.base_train import BaseTrain
from tqdm import tqdm
from utils.metrics import AverageMeter


class SfmNetTrainer(BaseTrain):
    def __init__(self, sess, model, config, logger, data_loader=None):
        super(SfmNetTrainer, self).__init__(sess, model, config, logger, data_loader)

        # load the model from the latest checkpoint
        self.model.load(self.sess)

    def train_epoch(self, epoch=None):
        """
        Train one epoch
        :param epoch: cur epoch number
        :return:
        """

        self.sess.run(self.data_loader.iterator.initializer,
                      feed_dict={self.data_loader.filenames: self.config.file_names})

        loss_per_epoch = AverageMeter()

        while True:
            try:
                loss = self.train_step()
            except tf.errors.OutOfRangeError:
                break

        loss_per_epoch.update(loss)
        self.sess.run(self.model.increment_cur_epoch_tensor)

        summaries_dict = {
            'train/loss_per_epoch': loss_per_epoch.val
        }

        self.logger.summarize(self.model.global_step_tensor.eval(self.sess), summaries_dict)
        # self.model.save(self.sess)

        print("""
Epoch={}  loss: {:.4f}
        """.format(epoch, loss_per_epoch.val))

    def train_step(self):
        """
        Run the session of train_step in tensorflow
        also get the loss of that minibatch.
        :return: tuple of some metrics to be used in summaries
        """
        feed_dict = {self.model.is_training: True}
        _, loss = self.sess.run([self.model.train_op, self.model.total_loss], feed_dict=feed_dict)
        f_frame_loss, b_frame_loss = self.sess.run([self.model.f_frame_loss, self.model.b_frame_loss])
        f_flow_sm_loss, b_flow_sm_loss = self.sess.run([self.model.f_flow_sm_loss, self.model.b_flow_sm_loss])
        f_motion_sm_loss, b_motion_sm_loss = self.sess.run([self.model.f_motion_sm_loss, self.model.b_motion_sm_loss])
        f_depth_loss, b_depth_loss = self.sess.run([self.model.f_depth_loss, self.model.b_depth_loss])

        self.sess.run(self.model.increment_global_step_tensor)

        summaries_dict = {
            'train/loss_per_step': loss,
            'train/f_frame_loss': f_frame_loss,
            'train/b_frame_loss': b_frame_loss,
            'train/f_flow_sm_loss': f_flow_sm_loss,
            'train/b_flow_sm_loss': b_flow_sm_loss,
            'train/f_motion_sm_loss': f_motion_sm_loss,
            'train/b_motion_sm_loss': b_motion_sm_loss,
            'train/f_depth_loss': f_depth_loss,
            'train/b_depth_loss': b_depth_loss
        }

        self.global_step = self.model.global_step_tensor.eval(self.sess)
        self.logger.summarize(self.global_step, summaries_dict)

        print("\tStep={} loss: {:.4f}".format(self.global_step, loss))

        if (self.global_step % 20) == 0:
            self.model.save(self.sess)

        return loss


if __name__ == "__main__":
    import tensorflow as tf
    from easydict import EasyDict
    import sys

    sys.path.extend(['..', '.'])

    from models.sfm_net_model import SfmNetModel
    from utils.logger import DefinedSummarizer
    from utils.dirs import create_dirs

    config = EasyDict({"image_height": 128,
                       "image_width": 384,
                       "num_channels": 3,
                       "learning_rate": 0.0003,
                       "beta1": 0.9,
                       "max_to_keep": 5,
                       "batch_size": 5,
                       "num_epochs": 5,
                       "data_loader": "SfmNetLoader_DeepTesla_TfRecords",
                       "file_names": ["../data/deeptesla/epochs/epoch01_front.tfrecord"],
                       "summary_dir": "../summary/",
                       "checkpoint_dir": "../checkpoint/",
                       "model_name": "sfm_net_deeptesla"
                       })
    create_dirs([config.summary_dir, config.checkpoint_dir])

    model = SfmNetModel(config)

    sess = tf.Session()
    logger = DefinedSummarizer(sess, config.summary_dir, scalar_tags=['train/loss_per_epoch',
                                                                      'train/loss_per_step',
                                                                      'train/f_frame_loss',
                                                                      'train/b_frame_loss',
                                                                      'train/f_flow_sm_loss',
                                                                      'train/b_flow_sm_loss',
                                                                      'train/f_depth_sm_loss',
                                                                      'train/f_motion_sm_loss',
                                                                      'train/b_motion_sm_loss',
                                                                      'train/f_depth_loss',
                                                                      'train/b_depth_loss'])
    trainer = SfmNetTrainer(sess, model, config, logger)
    trainer.train()
