import tensorflow as tf

W = tf.Variable([1.], dtype=tf.float32)
b = tf.Variable([-1.], dtype=tf.float32)
X = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

linear_model = tf.add(tf.multiply(W, X), b)
loss = tf.reduce_sum(tf.square(tf.subtract(linear_model, y)))

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(loss, feed_dict={X: [1, 2, 3, 4], y: [0.1, 1.1, 2.1, 3.1]}))
