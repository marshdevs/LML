import numpy as np
from six.moves import cPickle as pickle

class Data:
  def __init__(self):
    pass

  '''
  First reload the data we generated in assignment01
  '''
  def load_data(self, pickle_file):
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      train_dataset = save['train_dataset']
      train_labels = save['train_labels']
      valid_dataset = save['valid_dataset']
      valid_labels = save['valid_labels']
      test_dataset = save['test_dataset']
      test_labels = save['test_labels']
      del save  # hint to help gc free up memory
      print('Training set', train_dataset.shape, train_labels.shape)
      print('Validation set', valid_dataset.shape, valid_labels.shape)
      print('Test set', test_dataset.shape, test_labels.shape)
      return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

  '''
  Reformat into a shape that's more adapted to the models we're going to train:
    - data as a flat matrix
    - labels as float 1-hot encodings
  '''
  def reformat(self, dataset, labels, image_size, num_labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels
