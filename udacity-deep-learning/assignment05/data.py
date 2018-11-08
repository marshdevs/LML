import collections
import os
import zipfile
import tensorflow as tf
from six.moves.urllib.request import urlretrieve

class Data:
  def __init__(self):
    pass

  '''
  Download a file if not present, and make sure it's the right size.
  '''
  def maybe_download(self):
    url = 'http://mattmahoney.net/dc/'
    filename = 'text8.zip'
    expected_bytes = 31344016

    if not os.path.exists(filename):
      filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
      print('Found and verified %s' % filename)
    else:
      print(statinfo.st_size)
      raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

  '''
  Extract the first file enclosed in a zip file as a list of words, read the
  data into a string
  '''
  def read_data(self, filename):
    with zipfile.ZipFile(filename) as f:
      data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

  '''
  Build the dictionary and replace rare words with UNK token
  '''
  def build_dataset(self, words, vocabulary_size = 50000):
    count = [['UNK, -1']]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
      if word in dictionary:
        index = dictionary[word]
      else:
        index = 0 # dictionary['UNK']
        unk_count = unk_count + 1
      data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
