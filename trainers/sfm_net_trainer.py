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
        # initialize dataset
        self.sess.run(self.model.data_loader.initialize)
        # self.data_loader.initialize(self.sess, is_train=True)

        # initialize tqdm
        # loop = tqdm(range(self.data_loader.num_iterations), total=self.data_loader.num_iterations,
        #             desc="epoch-{}".format(epoch))
        loop = range(self.data_loader.num_iterations)

        loss_per_epoch = AverageMeter()

        for _ in loop:
            loss = self.train_step()
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

        # loop.close()

    def train_step(self):
        """
        Run the session of train_step in tensorflow
        also get the loss of that minibatch.
        :return: tuple of some metrics to be used in summaries
        """
        feed_dict = {self.model.is_training: True}
        try:
            _, loss = self.sess.run([self.model.train_step, self.model.total_loss], feed_dict=feed_dict)
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            self.sess.run(self.data_loader.initialize)
            _, loss = self.sess.run([self.model.train_step, self.model.total_loss], feed_dict=feed_dict)

        self.sess.run(self.model.increment_global_step_tensor)

        summaries_dict = {
            'train/loss_per_step': loss
        }
        global_step = self.model.global_step_tensor.eval(self.sess)
        self.logger.summarize(global_step, summaries_dict)

        print("""
    Step={} loss: {:.4f}
        """.format(global_step, loss))

        if (global_step % 20) == 0:
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
                       "data_loader": "SfmNetLoader_DeepTesla_SingleVideo",
                       "video_file": "../data/deeptesla/epochs/epoch01_front.mkv",
                       "summary_dir": "../summary/",
                       "checkpoint_dir": "../checkpoint/"
                       })
    create_dirs([config.summary_dir, config.checkpoint_dir])

    model = SfmNetModel(config)

    sess = tf.Session()
    logger = DefinedSummarizer(sess, config.summary_dir, scalar_tags=['train/loss_per_epoch', 'train/loss_per_step'])
    trainer = SfmNetTrainer(sess, model, config, logger)
    trainer.train()
