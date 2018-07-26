import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize
import imageio
import os
from keras import backend as K

imageio.plugins.ffmpeg.download()

f = 'epochs/epoch01_front.mkv'

try:
    os.getcwd()
    reader = imageio.get_reader(f)
    plt.imshow(reader.get_data(30))
    t0_frame, t1_frame, t2_frame, t3_frame = reader.get_data(0), reader.get_data(1), reader.get_data(
        2), reader.get_data(3)
    t0_frame = np.array(t0_frame)
    t1_frame = np.array(t1_frame)
    t2_frame = np.array(t2_frame)
    t3_frame = np.array(t3_frame)
except:
    pass


def img_pre_process(img):
    """
    Processes the image and returns it
    :param img: The image to be processed
    :return: Returns the processed image
    """
    # Chop off 1/3 from the top and cut bottom 150px(which contains the head of car)
    shape = img.shape
    img = img[int(shape[0] / 3):shape[0] - 150, 0:shape[1]]
    img = img / 255.
    print(img.shape)

    # Resize the image
    resize_img = resize(img, (128, 384), mode='reflect')
    # Return the image sized as a 4D array
    return resize_img  # np.resize(img, (w, h, c)


sess = tf.Session()

K.set_session(sess)


def model_input(img_h, img_w, img_c):
    I_t0 = tf.placeholder(tf.float32, (None, img_h, img_w, img_c), name='frame_t0')
    I_t1 = tf.placeholder(tf.float32, (None, img_h, img_w, img_c), name='frame_t1')
    learning_rate = tf.placeholder(tf.float32)
    steering = tf.placeholder(tf.float32, (None, 1), name='steering')

    return I_t0, I_t1, learning_rate, steering


img_h, img_w, img_c = 128, 384, 3
I_t0, I_t1, learning_rate, steering = model_input(img_h, img_w, img_c)
assert I_t0.shape.as_list() == [None, 128, 384, 3]
assert I_t1.shape.as_list() == [None, 128, 384, 3]
assert steering.shape.as_list() == [None, 1]



from structure import structure_net

f_point_cloud_1, f_depth_output = structure_net(I_t0)
b_point_cloud_1, b_depth_output = structure_net(I_t1, reuse=tf.AUTO_REUSE)
