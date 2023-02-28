import math
import numpy as np
from warnings import filterwarnings
filterwarnings("ignore")

def train_val_split(data, label, validation_split):
    """
    This function creates validation dataset to measure model performance.

    Args:
    - data (array): data to use in neural network.
    - label (array): target label to use in neural network.
    - validation_split (float): the fraction of the training data to use for validation.

    Returns:
    - train_data (array): data to use in model training.
    - train_label (array): target labels to use in model training.
    - val_data (array): data to use in model performance validating.
    - val_label (array): target label to use in in model performance validating.
    """
    # Calculate split size as integer
    split_size = int(len(data)*validation_split)
    
    # Create train data and label sets
    train_data = data[:split_size]
    train_label = label[:split_size]
    
    # Create validation data and label sets
    val_data = data[split_size:]
    val_label = label[split_size:]
    return train_data, train_label, val_data, val_label
  
def flatten(data):
    """
    This function flattens the input data.

    Args:
    - data (array): the input data to be flattened.

    Returns:
    - data (array): the flattened data.
    """
    data = data.reshape(data.shape[0], -1).T
    return data

def create_minibatches(data, label, batch_size, drop_remainder):
    """
    This function creates minibatches of data and labels for use in training the model.

    Args:
    - data (array): the input data to be split into minibatches.
    - label (array): the labels corresponding to the input data.
    - batch_size (int): the size of the minibatches used for training the model.
    - drop_remainder (bool): Whether to drop the last batch if it is smaller than the specified batch size.

    Returns:
    - minibatches (list of tuples): minibatches of preprocessed data and labels.
    """
    len_data = data.shape[1]
    minibatches = []

    # Shuffle the data and labels
    shuffled_indices = list(np.random.permutation(len_data))
    shuffled_data = data[:, shuffled_indices]
    shuffled_label = label[:, shuffled_indices].reshape((label.shape[0], len_data))

    # Create full minibatches
    full_minibatches = math.floor(len_data/batch_size)
    for i in range(full_minibatches):
        data_minibatch = shuffled_data[:, i*batch_size:(i+1)*batch_size]
        label_minibatch = shuffled_label[:, i*batch_size:(i+1)*batch_size]
        minibatches.append((data_minibatch, label_minibatch))

    # Create a minibatch with the remaining data if drop_remainder hyperparameter is False
    if drop_remainder==False:
      if len_data%batch_size != 0:
          data_minibatch = shuffled_data[:, full_minibatches*batch_size:]
          label_minibatch = shuffled_label[:, full_minibatches*batch_size:]
          minibatches.append((data_minibatch, label_minibatch))

    return minibatches

def preprocess(data, label, batch_size, drop_remainder):
    """
    This function runs flatten and create_minibatches functions together.

    Args:
    - data (array): input data.
    - label (array): input labels.
    - batch_size (int): the size of the minibatches used for training the model.
    - drop_remainder (bool): Whether to drop the last batch if it is smaller than the specified batch size.

    Returns:  
    - minibatches (list of tuple): tuple of input data and labels.
    """
    # Flatten data and label sets.
    data = flatten(data)
    label = flatten(label)

    # Create minibatches
    minibatches = create_minibatches(data, label, batch_size, drop_remainder)

    return minibatches