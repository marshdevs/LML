from data import Data

def main():
  data = Data()
  filename = data.maybe_download()
  words = data.read_data(filename)
  print('Data size: %d' % len(words))
  data, count, dictionary, reverse_dictionary = data.build_dataset(words)
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10])
  del words  # Hint to reduce memory.

if __name__ == '__main__': main()
