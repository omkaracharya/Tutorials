# Reference: https://pythonprogramming.net/tensorflow-neural-network-session-machine-learning-tutorial/

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Load MNIST Dataset
mnist_data = input_data.read_data_sets('/tmp/data', one_hot=True)

# Neural Network Structure
number_of_neurons_hidden_layer_1 = 500
number_of_neurons_hidden_layer_2 = 500
number_of_neurons_hidden_layer_3 = 500

# Number of classes and batch size
number_of_classes = 10
batch_size = 100

# For features and labels
X = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def build_neural_network_model(data):
    # This function does all the activation function calculations with data, weights and biases

    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([784, number_of_neurons_hidden_layer_1])),
                      'biases': tf.Variable(tf.random_normal([number_of_neurons_hidden_layer_1]))}
    hidden_layer_2 = {
        'weights': tf.Variable(tf.random_normal([number_of_neurons_hidden_layer_1, number_of_neurons_hidden_layer_2])),
        'biases': tf.Variable(tf.random_normal([number_of_neurons_hidden_layer_2]))}
    hidden_layer_3 = {
        'weights': tf.Variable(tf.random_normal([number_of_neurons_hidden_layer_2, number_of_neurons_hidden_layer_3])),
        'biases': tf.Variable(tf.random_normal([number_of_neurons_hidden_layer_3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([number_of_neurons_hidden_layer_3, number_of_classes])),
                    'biases': tf.Variable(tf.random_normal([number_of_classes]))}

    layer_1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    layer_3 = tf.nn.relu(layer_3)

    output = tf.matmul(layer_3, output_layer['weights']) + output_layer['biases']
    return output


def train_neural_network_model(X):
    # This function trains the neural network on training dataset with the help of Adam optimizer
    
    predictions = build_neural_network_model(X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    number_of_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(number_of_epochs):
            epoch_loss = 0
            for _ in range(int(mnist_data.train.num_examples / batch_size)):
                epoch_X, epoch_y = mnist_data.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={X: epoch_X, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', number_of_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({X: mnist_data.test.images, y: mnist_data.test.labels}))


def main():
    train_neural_network_model(X)


if __name__ == '__main__':
    main()
