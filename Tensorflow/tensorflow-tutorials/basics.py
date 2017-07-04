import tensorflow as tf
import numpy as np

# Variable for storing weights
W = tf.Variable([1.], dtype=tf.float32)

# Variable for storing biases
b = tf.Variable([-1.], dtype=tf.float32)

# Placeholder for training features
X = tf.placeholder(dtype=tf.float32)

# Placeholder for training labels
y = tf.placeholder(dtype=tf.float32)

# Definition of the linear model: W'X + b
linear_model = tf.add(tf.multiply(W, X), b)

# Loss in terms of 'Sum of Squared Errors'
loss = tf.reduce_sum(tf.square(tf.subtract(linear_model, y)))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss=loss)

# Training data
X_train = np.asarray([1., 2., 3., 4.])
y_train = np.asarray([0.1, 1.1, 2.1, 3.1])

number_of_iterations = 1000

# Tensorflow session
with tf.Session() as sess:
    # Initialize tf.Variable() instances
    init = tf.global_variables_initializer()

    # Run the tf session
    sess.run(init)

    # Training iterations
    for iteration in range(number_of_iterations):
        # Achieve the optimum point
        sess.run(train, feed_dict={X: X_train, y: y_train})

    # Get the weights, biases, and loss
    current_W, current_b, current_loss = sess.run([W, b, loss], feed_dict={X: X_train, y: y_train})

    # Print new weights, biases, and loss
    print("Weights: %s, Biases: %s, Loss: %s" % (current_W, current_b, current_loss))
