import os

import numpy as np

from keras import backend
from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file
from tensorflow.python.util.tf_export import keras_export

def get_img_class(tf_BatchDataset):
    data = tf_BatchDataset.unbatch()
    images = np.array(list(data.map(lambda x, y: x)))
    labels = np.array(list(data.map(lambda x, y: y)))
    print('NO.Images = ',len(images),'NO.Classes = ',len(labels),' Done!')
    return images , labels

def load_data():
  """Loads the CIFAR10 dataset.
  This is a dataset of 50,000 32x32 color training images and 10,000 test
  images, labeled over 10 categories. See more info at the
  [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html).
  The classes are:
  | Label | Description |
  |:-----:|-------------|
  |   0   | airplane    |
  |   1   | automobile  |
  |   2   | bird        |
  |   3   | cat         |
  |   4   | deer        |
  |   5   | dog         |
  |   6   | frog        |
  |   7   | horse       |
  |   8   | ship        |
  |   9   | truck       |
  Returns:
    Tuple of NumPy arrays: `(x_train, y_train), (x_test, y_test)`.
  **x_train**: uint8 NumPy array of grayscale image data with shapes
    `(50000, 32, 32, 3)`, containing the training data. Pixel values range
    from 0 to 255.
  **y_train**: uint8 NumPy array of labels (integers in range 0-9)
    with shape `(50000, 1)` for the training data.
  **x_test**: uint8 NumPy array of grayscale image data with shapes
    `(10000, 32, 32, 3)`, containing the test data. Pixel values range
    from 0 to 255.
  **y_test**: uint8 NumPy array of labels (integers in range 0-9)
    with shape `(10000, 1)` for the test data.
  Example:
  ```python
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
  assert x_train.shape == (50000, 32, 32, 3)
  assert x_test.shape == (10000, 32, 32, 3)
  assert y_train.shape == (50000, 1)
  assert y_test.shape == (10000, 1)
  ```
  """
#   dirname = 'cifar-10-batches-py'
#   origin = 'C:\Users\pv23228\Documents\Federated Learning POV\archive\chest_xray'
#   path = get_file(
#       dirname,
#       origin=origin,
#       untar=True,
#       file_hash=
#       '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce')

  # num_train_samples = 50000

  # x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
  # y_train = np.empty((num_train_samples,), dtype='uint8')

  # for i in range(1, 6):
  #   fpath = os.path.join(path, 'data_batch_' + str(i))
  #   (x_train[(i - 1) * 10000:i * 10000, :, :, :],
  #    y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

  # fpath = os.path.join(path, 'test_batch')
  # x_test, y_test = load_batch(fpath)

  # y_train = np.reshape(y_train, (len(y_train), 1))
  # y_test = np.reshape(y_test, (len(y_test), 1))

  # if backend.image_data_format() == 'channels_last':
  #   x_train = x_train.transpose(0, 2, 3, 1)
  #   x_test = x_test.transpose(0, 2, 3, 1)

  # x_test = x_test.astype(x_train.dtype)
  # y_test = y_test.astype(y_train.dtype)

  BATCH_SIZE = 64
  IMAGE_SIZE = 224
  train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
      r"C:\Users\pv23228\Documents\Federated Learning POV\archive\chest_xray\train",
      shuffle = True,
      image_size = (IMAGE_SIZE,IMAGE_SIZE),
      batch_size = BATCH_SIZE 
  )
  test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
      r"C:\Users\pv23228\Documents\Federated Learning POV\archive\chest_xray\test",
      shuffle = True,
      image_size = (IMAGE_SIZE,IMAGE_SIZE),
      batch_size = BATCH_SIZE 
  )
  val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
      r"C:\Users\pv23228\Documents\Federated Learning POV\archive\chest_xray\val",
      shuffle = True,
      image_size = (IMAGE_SIZE,IMAGE_SIZE),
      batch_size = BATCH_SIZE 
  )

  images_train , classes_train = get_img_class(train_dataset)
  images_test , classes_test = get_img_class(test_dataset)
  images_val , classes_val = get_img_class(val_dataset)

  # Normalize the data. Before we need to connvert data type to float for computation.
  x_train = images_train.astype('float32')
  x_test = images_test.astype('float32')
  x_train /= 255
  x_test /= 255
  num_classes = 2
  # Convert class vectors to binary class matrices. This is called one hot encoding.
  y_train = utils.np_utils.to_categorical(classes_train, num_classes)
  y_test = utils.np_utils.to_categorical(classes_test, num_classes)

  return (x_train, y_train), (x_test, y_test)
