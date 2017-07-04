import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Get MNIST data from tensorflow examples
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Placeholder for training features
# The image size is 28*28 = 784 pixels
# None - any number of examples
X = tf.placeholder(tf.float32, [None, 784])

# Variable for weights
# Dimensions: feature dim * number of classes
W = tf.Variable(tf.zeros([784, 10]))

# Variable for biases
# Dimensions: number of classes
b = tf.Variable(tf.zeros([10]))

# Class labels using linear regression
y = tf.add(tf.matmul(X, W), b)

# Actual class labels
y_ = tf.placeholder(tf.float32, [None, 10])

# Cross entropy as loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Gradient Descent for linear regression
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entropy)

number_of_iterations = 1000

# Tensorflow session
with tf.Session() as sess:
    # Initialize tf variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training iterations
    for iteration in range(number_of_iterations):
        # Batch of 100 images
        batch_X, batch_y = mnist.train.next_batch(100)

        # Gradient Descent
        sess.run(train, feed_dict={X: batch_X, y_: batch_y})

    # Correct predictions
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    # Accuracy computation
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # Testing accuracy
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, y_: mnist.test.labels}))
