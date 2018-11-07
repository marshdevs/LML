from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import range
from data import Data

class Tensor:
  def __init__(self, image_size, num_labels, train_subset, train_dataset,
    train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    self.image_size = image_size
    self.num_labels = num_labels
    self.train_subset = train_subset
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

  '''
  Let's load all the data into TensorFlow and build the computation graph corresponding to our training:
  '''
  def build_graph(self):
    with self.graph.as_default():
      # Input data.
      # Load the training, validation and test data into constants that are
      # attached to the graph.
      tf_train_dataset = tf.constant(self.train_dataset[:self.train_subset, :])
      tf_train_labels = tf.constant(self.train_labels[:self.train_subset])
      tf_valid_dataset = tf.constant(self.valid_dataset)
      tf_test_dataset = tf.constant(self.test_dataset)

      # Variables.
      # These are the parameters that we are going to be training. The weight
      # matrix will be initialized using random values following a (truncated)
      # normal distribution. The biases get initialized to zero.
      weights = tf.Variable(tf.truncated_normal([self.image_size * self.image_size, self.num_labels]))
      biases = tf.Variable(tf.zeros([self.num_labels]))

      # Training computation.
      # We multiply the inputs with the weight matrix, and add biases. We compute
      # the softmax and cross-entropy (it's one operation in TensorFlow, because
      # it's very common, and it can be optimized). We take the average of this
      # cross-entropy across all training examples: that's our loss.
      logits = tf.matmul(tf_train_dataset, weights) + biases
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

      # Optimizer.
      # We are going to find the minimum of this loss using gradient descent.
      self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

      # Predictions for the training, validation, and test data.
      # These are not part of training, but merely here so that we can report
      # accuracy figures as we train.
      self.train_prediction = tf.nn.softmax(logits)
      self.valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
      self.test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

  '''
  Let's run this computation and iterate:
  '''
  def run_graph(self, num_steps):
    with tf.Session(graph=self.graph) as session:
      # This is a one-time operation which ensures the parameters get initialized as
      # we described in the graph: random weights for the matrix, zeros for the
      # biases.
      tf.global_variables_initializer().run()
      print('Initialized')
      for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        _, l, predictions = session.run([self.optimizer, self.loss, self.train_prediction])
        if (step % 100 == 0):
          print('Loss at step %d: %f' % (step, l))
          print('Training accuracy: %.1f%%' % self.accuracy(predictions, self.train_labels[:self.train_subset, :]))
          # Calling .eval() on valid_prediction is basically like calling run(), but
          # just to get that one numpy array. Note that it recomputes all its graph
          # dependencies.
          print('Validation accuracy: %.1f%%' % self.accuracy(self.valid_prediction.eval(), self.valid_labels))
      print('Test accuracy: %.1f%%' % self.accuracy(self.test_prediction.eval(), self.test_labels))

  '''
  Let's now switch to stochastic gradient descent training instead, which is
  much faster. The graph will be similar, except that instead of holding all the
  training data into a constant node, we create a Placeholder node which will be
  fed actual data at every call of session.run().
  '''
  def build_stochastic(self, batch_size):
    with self.graph.as_default():
      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, self.image_size * self.image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, self.num_labels))
      tf_valid_dataset = tf.constant(self.valid_dataset)
      tf_test_dataset = tf.constant(self.test_dataset)

      # Variables.
      weights = tf.Variable(tf.truncated_normal([self.image_size * self.image_size, self.num_labels]))
      biases = tf.Variable(tf.zeros([self.num_labels]))

      # Training computation.
      logits = tf.matmul(tf_train_dataset, weights) + biases
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

      # Optimizer.
      self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

      # Predictions for the training, validation, and test data.
      self.train_prediction = tf.nn.softmax(logits)
      self.valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
      self.test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)
      return tf_train_dataset, tf_train_labels

  def run_stochastic(self, batch_size, num_steps, tf_train_dataset, tf_train_labels):
    with tf.Session(graph=self.graph) as session:
      tf.global_variables_initializer().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (self.train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = self.train_dataset[offset:(offset + batch_size), :]
        batch_labels = self.train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % self.accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % self.accuracy(self.valid_prediction.eval(), self.valid_labels))
      print("Test accuracy: %.1f%%" % self.accuracy(self.test_prediction.eval(), self.test_labels))

  def build_relu(self, batch_size, hidden_nodes):
    with self.graph.as_default():
      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, self.image_size * self.image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, self.num_labels))
      tf_valid_dataset = tf.constant(self.valid_dataset)
      tf_test_dataset = tf.constant(self.test_dataset)

      # Variables.
      weights1 = tf.Variable(tf.truncated_normal([self.image_size * self.image_size, hidden_nodes]))
      biases1 = tf.Variable(tf.zeros([hidden_nodes]))
      weights2 = tf.Variable(tf.truncated_normal([hidden_nodes, self.num_labels]))
      biases2 = tf.Variable(tf.zeros([self.num_labels]))

      # Training computation.
      logits = self.forward_prop(tf_train_dataset, weights1, biases1, weights2, biases2)
      self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))

      # Optimizer.
      self.optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

      # Predictions for the training, validation, and test data.
      self.train_prediction = tf.nn.softmax(logits)
      self.valid_prediction = tf.nn.softmax(self.forward_prop(tf_valid_dataset, weights1, biases1, weights2, biases2))
      self.test_prediction = tf.nn.softmax(self.forward_prop(tf_test_dataset, weights1, biases1, weights2, biases2))
      return tf_train_dataset, tf_train_labels

  def forward_prop(self, inputs, weights1, biases1, weights2, biases2):
    return tf.matmul(tf.nn.relu(tf.matmul(inputs, weights1)) + biases1, weights2) + biases2

  def accuracy(self, predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def main():
  data = Data()
  pickle_file = '../assignment01/data/notMNIST.pickle'
  train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = data.load_data(pickle_file)

  image_size = 28
  num_labels = 10
  train_dataset, train_labels = data.reformat(train_dataset, train_labels, image_size, num_labels)
  valid_dataset, valid_labels = data.reformat(valid_dataset, valid_labels, image_size, num_labels)
  test_dataset, test_labels = data.reformat(test_dataset, test_labels, image_size, num_labels)
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

  train_subset = 10000
  num_steps = 3001
  batch_size = 128
  hidden_nodes = 1024
  tensor = Tensor(image_size, num_labels, train_subset, train_dataset,
    train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
  # tensor.build_graph()
  # tensor.run_graph(num_steps)
  # tf_train_dataset, tf_train_labels = tensor.build_stochastic(batch_size)
  # tensor.run_stochastic(batch_size, num_steps, tf_train_dataset, tf_train_labels)
  tf_train_dataset, tf_train_labels = tensor.build_relu(batch_size, hidden_nodes)
  tensor.run_stochastic(batch_size, num_steps, tf_train_dataset, tf_train_labels)


if __name__ == "__main__": main()
