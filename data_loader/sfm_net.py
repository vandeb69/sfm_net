from easydict import EasyDict
import tensorflow as tf
import imageio
from skimage.transform import resize


class VideoReader:
    def __init__(self, video_file, start, preprocesser):
        self.preprocesser = preprocesser
        self.reader = imageio.get_reader(video_file)
        for _ in range(start):
            _ = self.reader.get_next_data()
        self.max = self.reader.get_length() - 3
        self.i = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.i <= self.max:
            frame = self.reader.get_next_data()
            frame = self.preprocesser(frame)
            self.i += 1
            return frame
        else:
            self.reader.close()
            raise StopIteration


class SfmNetLoader_DeepTesla():
    def __init__(self, config):
        self.config = config
        self.frames_t0 = VideoReader(self.config.video_file, start=0, preprocesser=self.DeepTesla_preprocess)
        self.frames_t1 = VideoReader(self.config.video_file, start=1, preprocesser=self.DeepTesla_preprocess)
        self.generator_t0 = self.frames_t0.__iter__
        self.generator_t1 = self.frames_t1.__iter__

        self.dataset = None
        self.iterator = None
        self.initialize = None
        self.saveable = None
        self.next_batch = None

        self.build_dataset_operator()

    @staticmethod
    def DeepTesla_preprocess(img):
        """
        Processes the image and returns it
        :param img: The image to be processed
        :return: Returns the processed image
        """
        shape = img.shape
        img = img[int(shape[0] / 3):shape[0] - 150, 0:shape[1]]
        img = img / 255.

        resize_img = resize(img, (128, 384), mode='reflect')
        return resize_img

    def build_dataset_operator(self):
        img_h = self.config.image_height
        img_w = self.config.image_width
        img_c = self.config.num_channels

        dataset_t0 = tf.data.Dataset.from_generator(self.generator_t0, tf.float32,
                                                    tf.TensorShape([img_h, img_w, img_c]))
        dataset_t1 = tf.data.Dataset.from_generator(self.generator_t1, tf.float32,
                                                    tf.TensorShape([img_h, img_w, img_c]))
        dataset = tf.data.Dataset.zip((dataset_t0, dataset_t1))
        self.dataset = dataset.batch(self.config.batch_size)
        self.iterator = self.dataset.make_initializable_iterator()
        self.initialize = self.iterator.initializer
        self.saveable = tf.contrib.data.make_saveable_from_iterator(self.iterator)
        self.next_batch = self.iterator.get_next()


if __name__ == "__main__":
    config = EasyDict({"video_file": "../data/deeptesla/epochs/epoch01_front.mkv",
                       "batch_size": 10,
                       "image_height": 128,
                       "image_width": 384,
                       "num_channels": 3})
    loader = SfmNetLoader_DeepTesla(config)

    sess = tf.Session()
    sess.run(loader.initialize)
    while True:
        try:
            I_t0, I_t1 = sess.run(loader.next_batch)
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            break