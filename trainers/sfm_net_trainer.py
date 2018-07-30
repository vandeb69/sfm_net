from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class SfmNetTrainer(BaseTrain):
    def __init__(self, sess, model, config, logger, data_loader):
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
        self.data_loader.initialize(self.sess, is_train=True)

        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        for _ in loop:
            loss = self.train_step()
            losses.append(loss)
        loss = np.mean(losses)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'total_loss': loss
        }

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)


    def train_step(self):
        I_t0, I_t1 = next(self.data.next_batch())
        feed_dict = {self.model.I_t0: I_t0, self.model.I_t1: I_t1, self.model.is_training: True}
        _, loss = self.sess.run([self.model.train_step, self.model.total_loss], feed_dict=feed_dict)

        return loss