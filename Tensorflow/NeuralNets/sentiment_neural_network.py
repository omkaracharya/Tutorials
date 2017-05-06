# Reference: https://pythonprogramming.net/tensorflow-neural-network-session-machine-learning-tutorial/

import numpy as np
import tensorflow as tf
from sentiment_features import create_data

train_X, validation_X, train_Y, validation_Y = create_data('../data/pos.txt', '../data/neg.txt')

# Neural Network Structure
number_of_neurons_hidden_layer_1 = 500
number_of_neurons_hidden_layer_2 = 500
number_of_neurons_hidden_layer_3 = 500

# Number of classes and batch size
number_of_classes = 2
batch_size = 100

# For features and labels
X = tf.placeholder('float', [None, len(train_X[0])])
y = tf.placeholder('float')


def build_neural_network_model(data):
    # This function does all the activation function calculations with data, weights and biases

    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([len(train_X[0]), number_of_neurons_hidden_layer_1])),
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

            i = 0
            while i < len(train_X):
                start = i
                end = i + batch_size
                batch_X = np.array(train_X[start:end])
                batch_Y = np.array(train_Y[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_X, y: batch_Y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch + 1, 'completed out of', number_of_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({X: validation_X, y: validation_Y}))


def main():
    train_neural_network_model(X)


if __name__ == '__main__':
    main()
