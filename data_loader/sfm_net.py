from easydict import EasyDict
import tensorflow as tf
import imageio
from skimage.transform import resize
import skvideo.io


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def video2tfrecord_sfmnet_deeptesla(video_file, file_name):

    video_data = skvideo.io.vread(video_file)
    shape = video_data.shape
    video_data = video_data[:, int(shape[1] / 3):shape[1] - 150, 0:shape[2], :]

    frames_t0 = video_data[:-1]
    frames_t1 = video_data[1:]

    zipped_frames = zip(frames_t0, frames_t1)

    writer = tf.python_io.TFRecordWriter(file_name)

    for frame_t0, frame_t1 in zipped_frames:
        height = frame_t0.shape[0]
        width = frame_t0.shape[1]

        frame_t0_raw = tf.compat.as_bytes(frame_t0.tostring())
        frame_t1_raw = tf.compat.as_bytes(frame_t1.tostring())

        # create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'frame_t0_raw': _bytes_feature(frame_t0_raw),
            'frame_t1_raw': _bytes_feature(frame_t1_raw)
        }))

        # serialize to string and write on the tfrecord file
        writer.write(example.SerializeToString())

    writer.close()


class SfmNetLoader_DeepTesla_TfRecords():
    def __init__(self, config):
        self.config = config
        self.filenames = tf.placeholder(tf.string, shape=[None])

        self.dataset = None
        self.iterator = None
        self.next_batch = None

        self.build_dataset_op()

    def build_dataset_op(self):
        dataset = tf.data.TFRecordDataset(self.filenames)
        dataset = dataset.map(self._parse_function)
        self.dataset = dataset.batch(self.config.batch_size)#.shuffle(buffer_size=10000)#.repeat()
        self.iterator = self.dataset.make_initializable_iterator()
        self.next_batch = self.iterator.get_next()

    @staticmethod
    def _parse_function(example_proto):
        keys_to_features = {'height': tf.FixedLenFeature([], tf.int64),
                            'width': tf.FixedLenFeature([], tf.int64),
                            'frame_t0_raw': tf.FixedLenFeature([], tf.string),
                            'frame_t1_raw': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        height = tf.cast(parsed_features['height'], tf.int32)
        width = tf.cast(parsed_features['width'], tf.int32)

        frame_shape = tf.stack([height, width, 3])

        frame_t0 = tf.decode_raw(parsed_features['frame_t0_raw'], tf.uint8)
        frame_t0 = tf.reshape(frame_t0, frame_shape)
        frame_t0 = tf.cast(frame_t0, tf.float32) / 255.
        frame_t0 = tf.image.resize_images(frame_t0, [128, 384])

        frame_t1 = tf.decode_raw(parsed_features['frame_t1_raw'], tf.uint8)
        frame_t1 = tf.reshape(frame_t1, frame_shape)
        frame_t1 = tf.cast(frame_t1, tf.float32) / 255.
        frame_t1 = tf.image.resize_images(frame_t1, [128, 384])

        return frame_t0, frame_t1


class VideoReader:
    def __init__(self, video_file, start, preprocesser):
        self.preprocesser = preprocesser
        self.reader = imageio.get_reader(video_file)
        for _ in range(start):
            _ = self.reader.get_next_data()
        self.length = self.reader.get_length() - 3
        self.i = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.i <= self.length:
            frame = self.reader.get_next_data()
            frame = self.preprocesser(frame)
            self.i += 1
            return frame
        else:
            self.reader.close()
            raise StopIteration


class SfmNetLoader_DeepTesla_SingleVideo():
    def __init__(self, config):
        self.config = config

        self.frames_t0 = None
        self.frames_t1 = None
        self.generator_t0 = None
        self.generator_t1 = None
        self.num_iterations = None
        self.dataset = None
        self.iterator = None
        self.saveable = None
        self.next_batch = None

        self.build_dataset_op()

    def build_dataset_op(self, start=0):
        self.frames_t0 = VideoReader(self.config.video_file, start=start, preprocesser=self.DeepTesla_preprocess)
        self.frames_t1 = VideoReader(self.config.video_file, start=start + 1, preprocesser=self.DeepTesla_preprocess)
        self.generator_t0 = self.frames_t0.__iter__
        self.generator_t1 = self.frames_t1.__iter__

        self.num_iterations = (self.frames_t0.length + self.config.batch_size - 1) // self.config.batch_size

        img_h = self.config.image_height
        img_w = self.config.image_width
        img_c = self.config.num_channels

        dataset_t0 = tf.data.Dataset.from_generator(self.generator_t0, tf.float32,
                                                    tf.TensorShape([img_h, img_w, img_c]))
        dataset_t1 = tf.data.Dataset.from_generator(self.generator_t1, tf.float32,
                                                    tf.TensorShape([img_h, img_w, img_c]))
        self.dataset = tf.data.Dataset.zip((dataset_t0, dataset_t1))
        # self.dataset = dataset.batch(self.config.batch_size).repeat()
        self.iterator = self.dataset.make_one_shot_iterator()
        # self.saveable = tf.contrib.data.make_saveable_from_iterator(self.iterator)
        # tf.add_to_collection(tf.GraphKeys.SAVEABLE_OBJECTS, self.saveable)
        self.next_batch = self.iterator.get_next()

        print("Data loaded successfully..")

    @staticmethod
    def DeepTesla_preprocess(img):
        """
        Processes the image and returns it
        :param img: The image to be processed
        :return: Returns the processed image
        """
        shape = img.shape
        img = img[int(shape[0] / 3):shape[0] - 150, 0:shape[1]]
        # img = img / 255.

        resize_img = resize(img, (128, 384), mode='reflect')
        return resize_img


if __name__ == "__main__":

    config = EasyDict({"video_file": "../data/deeptesla/epochs/epoch01_front.mkv",
                       "batch_size": 100,
                       "image_height": 128,
                       "image_width": 384,
                       "num_channels": 3})

    video2tfrecord_sfmnet_deeptesla(config.video_file, "../data/deeptesla/epochs/epoch01_front.tfrecord")

    # record_iterator = tf.python_io.tf_record_iterator(path="../data/deeptesla/epochs/epoch01_front.tfrecord")
    # for string_record in record_iterator:
    #     example = tf.train.Example()
    #     example.ParseFromString(string_record)
    #
    #     height = (example.features.feature['height'].int64_list.value)
    #     width = (example.features.feature['width'].int64_list.value)
    #     frame_t0 = (example.features.feature['frame_t0_raw'].bytes_list.value)
    #     frame_t1 = (example.features.feature['frame_t0_raw'].bytes_list.value)

    loader = SfmNetLoader_DeepTesla_TfRecords(config)

    sess = tf.Session()

    sess.run(loader.iterator.initializer,
             feed_dict={loader.filenames: ["../data/deeptesla/epochs/epoch01_front.tfrecord"]})
    while True:
        try:
            I_t0, I_t1 = sess.run(loader.next_batch)
            print("\t", I_t0.shape, I_t1.shape)
        except tf.errors.OutOfRangeError:
            print("End of dataset")
            print("Reinitializing...")
            sess.run(loader.iterator.initializer,
                     feed_dict={loader.filenames: ["../data/deeptesla/epochs/epoch01_front.tfrecord"]})
            continue


