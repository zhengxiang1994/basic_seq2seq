import tensorflow as tf
import numpy as np

a = np.array([[1, 2], [5, 3], [2, 6]])
b = tf.Variable(a)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(b))
    print(sess.run(tf.reduce_max(b, axis=None)))
