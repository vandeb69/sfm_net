import numpy as np
import tensorflow as tf
import keras.backend as K

from structure import structure_net
from motion import motion_net
from loss import get_frame_loss, get_smooth_loss, get_fb_depth_loss


def model_input(img_h, img_w, img_c):
    I_t0 = tf.placeholder(tf.float32, (None, img_h, img_w, img_c), name='frame_t0')
    I_t1 = tf.placeholder(tf.float32, (None, img_h, img_w, img_c), name='frame_t1')
    learning_rate = tf.placeholder(tf.float32)
    steering = tf.placeholder(tf.float32, (None, 1), name='steering')

    return I_t0, I_t1, learning_rate, steering


sess = tf.Session()
K.set_session(sess)

img_h, img_w, img_c = 128, 384, 3
I_t0, I_t1, learning_rate, steering = model_input(img_h, img_w, img_c)
assert I_t0.shape.as_list() == [None, 128, 384, 3]
assert I_t1.shape.as_list() == [None, 128, 384, 3]
assert steering.shape.as_list() == [None, 1]

f_point_cloud_1, f_depth_output = structure_net(I_t0)
b_point_cloud_1, b_depth_output = structure_net(I_t1, reuse=tf.AUTO_REUSE)

f_pix_pos, f_flow, f_point_cloud_2, f_motion_map = motion_net(I_t0, I_t1, f_point_cloud_1)
b_pix_pos, b_flow, b_point_cloud_2, b_motion_map = motion_net(I_t1, I_t0, b_point_cloud_1, reuse=tf.AUTO_REUSE)

f_frame_loss = get_frame_loss()(I_t0, I_t1, f_pix_pos)
b_frame_loss = get_frame_loss()(I_t1, I_t0, b_pix_pos, reuse=tf.AUTO_REUSE)

f_flow_sm_loss = get_smooth_loss(order=1)(f_flow, 'flow')
b_flow_sm_loss = get_smooth_loss(order=1)(b_flow, 'flow', reuse=tf.AUTO_REUSE)

f_depth_sm_loss = get_smooth_loss(order=1)(f_depth_output, 'depth')
b_depth_sm_loss = get_smooth_loss(order=1)(b_depth_output, 'depth', reuse=tf.AUTO_REUSE)

f_motion_sm_loss = get_smooth_loss(order=1)(f_motion_map, 'motion')
b_motion_sm_loss = get_smooth_loss(order=1)(b_motion_map, 'motion', reuse=tf.AUTO_REUSE)

f_depth_loss = get_fb_depth_loss()(f_depth_output, b_depth_output, f_pix_pos, f_point_cloud_1)
b_depth_loss = get_fb_depth_loss()(b_depth_output, f_depth_output, b_pix_pos, b_point_cloud_1, reuse=tf.AUTO_REUSE)

total_loss = f_depth_loss + f_depth_sm_loss + f_motion_sm_loss + f_flow_sm_loss + f_frame_loss +\
    b_depth_loss + b_depth_sm_loss + b_motion_sm_loss + b_flow_sm_loss + b_frame_loss

train_op = tf.train.AdamOptimizer(learning_rate=0.0003, beta1=0.9).minimize(total_loss)

tf.summary.scalar('total_loss', total_loss)
tf.summary.image('b_depth_output', b_depth_output, 1)
tf.summary.image('f_depth_output', f_depth_output, 1)

merger_summary = tf.summary.merge_all()

write = tf.summary.FileWriter('/tmp/sfm')
write.add_graph(sess.graph)


test_frame0 = np.load('src/test_frame0.npy')
test_frame1 = np.load('src/test_frame1.npy')

sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for step in range(10000):
    ckpt = tf.train.get_checkpoint_state('./checkpoint/')
    if (step == 0) and ckpt and ckpt.model_checkpoint_path:
        print('load model')
        saver.restore(sess, ckpt.model_checkpoint_path)

    print(step)

    feed_dict = {I_t0: test_frame0, I_t1: test_frame1}
    _ = sess.run([train_op], feed_dict=feed_dict)

    if step % 5 == 0:
        train_loss = sess.run(total_loss, feed_dict=feed_dict)
        print(train_loss)

    if step % 100 == 0:
        s = sess.run(merger_summary, feed_dict=feed_dict)
        write.add_summary(s, step)
        saver.save(sess, './checkpoint/' + 'my-model', global_step=step)