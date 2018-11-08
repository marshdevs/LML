from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import range
from data import Data

'''
**Assignment 4**

Previously in assignment02 and assignment03, we trained fully connected
networks to classify notMNIST characters.

The goal of this assignment is make the neural network convolutional.
'''

'''
Let's build a small network with two convolutional layers, followed by one
fully connected layer. Convolutional networks are more expensive
computationally, so we'll limit its depth and number of fully connected nodes.
'''
class Tensor:
  def __init__(self, image_size, num_labels, num_channels, train_dataset,
    train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    self.image_size = image_size
    self.num_labels = num_labels
    self.num_channels = num_channels
    self.train_dataset = train_dataset
    self.train_labels = train_labels
    self.valid_dataset = valid_dataset
    self.valid_labels = valid_labels
    self.test_dataset = test_dataset
    self.test_labels = test_labels
    self.graph = tf.Graph()
    self.loss = None
    self.optimizer = None
    self.train_prediction = None
    self.valid_prediction = None
    self.test_prediction = None

  def build_graph(self, batch_size, patch_size, depth, num_hidden):
    with self.graph.as_default():
      # Input data
      tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, self.image_size, self.image_size, self.num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, self.num_labels))
      tf_valid_dataset = tf.constant(self.valid_dataset)
      tf_test_dataset = tf.constant(self.test_dataset)

      # Variables
      layer1_weights = tf.Variable(tf.truncated_normal(
        [patch_size, patch_size, self.num_channels, depth], stddev=0.1))
      layer1_biases = tf.Variable(tf.zeros([depth]))
      layer2_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
      layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
      layer3_weights = tf.Variable(tf.truncated_normal(
          [self.image_size // 4 * self.image_size // 4 * depth, num_hidden], stddev=0.1))
      layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
      layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, self.num_labels], stddev=0.1))
      layer4_biases = tf.Variable(tf.constant(1.0, shape=[self.num_labels]))

      # Training computation
      weights = [layer1_weights, layer2_weights, layer3_weights, layer4_weights]
      biases = [layer1_biases, layer2_biases, layer3_biases, layer4_biases]
      logits = self.model(tf_train_dataset, weights, biases)
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

      # Optimizer
      self.optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(self.loss)

      # Predictions for training, validation, and test data
      self.train_prediction = tf.nn.softmax(logits)
      self.valid_prediction = tf.nn.softmax(self.model(tf_valid_dataset, weights, biases))
      self.test_prediction = tf.nn.softmax(self.model(tf_test_dataset, weights, biases))
      return tf_train_dataset, tf_train_labels

  # Network model
  def model(self, data, weights, biases):
    conv = tf.nn.conv2d(data, weights[0], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + biases[0])
    max_pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    conv = tf.nn.conv2d(max_pool, weights[1], [1, 1, 1, 1], padding='SAME')
    hidden = tf.nn.relu(conv + biases[1])
    max_pool = tf.nn.max_pool(hidden, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    shape = max_pool.get_shape().as_list()
    reshape = tf.reshape(max_pool, [shape[0], shape[1] * shape[2] * shape[3]])
    hidden = tf.nn.relu(tf.matmul(reshape, weights[2]) + biases[2])
    return tf.matmul(hidden, weights[3]) + biases[3]

  def run(self, num_steps, batch_size, tf_train_dataset, tf_train_labels):
    with tf.Session(graph=self.graph) as session:
      tf.global_variables_initializer().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * batch_size) % (self.train_labels.shape[0] - batch_size)
        batch_data = self.train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = self.train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
          [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Validation accuracy: %.1f%%' % accuracy(self.valid_prediction.eval(), self.valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(self.test_prediction.eval(), self.test_labels))

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def main():
  image_size = 28
  num_labels = 10
  num_channels = 1 # grayscale
  pickle_file = '../assignment01/data/notMNIST.pickle'
  data = Data()
  train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data.load_data(pickle_file)
  train_dataset, train_labels = data.reformat(train_dataset, train_labels, image_size, num_labels, num_channels)
  valid_dataset, valid_labels = data.reformat(valid_dataset, valid_labels, image_size, num_labels, num_channels)
  test_dataset, test_labels = data.reformat(test_dataset, test_labels, image_size, num_labels, num_channels)
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

  batch_size = 16
  patch_size = 5
  depth = 16
  num_hidden = 64
  num_steps = 1001
  tensor = Tensor(image_size, num_labels, num_channels, train_dataset,
    train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
  tf_train_dataset, tf_train_labels = tensor.build_graph(batch_size, patch_size, depth, num_hidden)
  tensor.run(num_steps, batch_size, tf_train_dataset, tf_train_labels)

if __name__ == "__main__": main()
