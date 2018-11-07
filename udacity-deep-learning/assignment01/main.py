from get_data import GetData
from sklearn.linear_model import LogisticRegression

class Classifier:
  def __init__(self):
    pass

def run_sklearn_classifier(train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels):
  train_dataset = train_dataset.reshape(len(train_dataset), -1)
  test_dataset = test_dataset.reshape(len(test_dataset), -1)
  valid_dataset = valid_dataset.reshape(len(valid_dataset), -1)

  clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(train_dataset[:5000], train_labels[:5000])
  print(clf.predict(test_dataset))
  print(clf.predict_proba(test_dataset))
  print(clf.score(test_dataset, test_labels))

def main():
  image_size = 28  # Pixel width and height.
  pixel_depth = 255.0  # Number of levels per pixel.
  data = GetData()
  train_filename = data.maybe_download('notMNIST_large.tar.gz', 247336696)
  test_filename = data.maybe_download('notMNIST_small.tar.gz', 8458043)
  train_folders = data.maybe_extract(train_filename)
  test_folders = data.maybe_extract(test_filename)
  train_datasets = data.maybe_pickle(train_folders, image_size, pixel_depth, 45000)
  test_datasets = data.maybe_pickle(test_folders, image_size, pixel_depth, 1800)

  train_size = 200000
  valid_size = 10000
  test_size = 10000
  valid_dataset, valid_labels, train_dataset, train_labels = data.merge_datasets(
    train_datasets, train_size, image_size, valid_size)
  _, _, test_dataset, test_labels = data.merge_datasets(test_datasets, test_size, image_size)

  print('Training:', train_dataset.shape, train_labels.shape)
  print('Validation:', valid_dataset.shape, valid_labels.shape)
  print('Testing:', test_dataset.shape, test_labels.shape)

  train_dataset, train_labels = data.randomize(train_dataset, train_labels)
  test_dataset, test_labels = data.randomize(test_dataset, test_labels)
  valid_dataset, valid_labels = data.randomize(valid_dataset, valid_labels)

  print('Compressed pickle size:', data.save_data(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels))

  # Run a logistic classifier on the training data
  run_sklearn_classifier(train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels)

if __name__ == "__main__": main()
