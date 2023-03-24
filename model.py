import numpy as np
import initializer
import activations
from metrics import compute_cost, accuracy_score
from load_dataset import train_val_split, preprocess
from neural_network import NetworkPropagation
from update_gradients import UpdateGradient

class DNNModel:
  """
  This class builds and trains a neural network model.
  """
  def __init__(self, layers, hidden_layer_activation="relu", output_layer_activation="softmax", kernel_initializer="he", 
              loss_type="categorical_crossentropy", dropout_rate=0., lambd=0., 
              optimizer="adam", learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
      """
      Args:
      - layers (list): a list of integers representing the number of neurons in each layer of the neural network.
      - hidden_layer_activation (str, optional): the activation function for the hidden layers of the neural network.
           Defaults to "relu".
      - output_layer_activation (str, optional): the activation function for the output layer of the neural network.
           Defaults to "softmax".
      - kernel_initializer (str, optional): the initialization method for the neural network weights. Defaults to "he".
      - loss_type (str, optional): the loss function used during training. Defaults to "categorical_crossentropy".
      - dropout_rate (float, optional): the dropout rate used during training. Defaults to 0.0.
      - lambd (float, optional): the L2 regularization penalty used during training. Defaults to 0.0.
      - optimizer (str, optional): the optimization algorithm used during training. Defaults to "adam".
      - learning_rate (float, optional): the learning rate used during training. Defaults to 0.001.
      - beta1 (float, optional): the exponential decay rate for Adam first moment estimates or Momentum.
           Defaults to 0.9.
      - beta2 (float, optional): the exponential decay rate for Adam second moment estimates or RMSprop.
           Defaults to 0.999.
      - epsilon (float, optional): a small number used to avoid division by zero. Defaults to 1e-8.
      """ 
      self.layers = layers
      self.hidden_layer_activation = hidden_layer_activation
      self.output_layer_activation = output_layer_activation
      self.kernel_initializer = kernel_initializer
      self.loss_type = loss_type
      self.dropout_rate = dropout_rate
      self.lambd = lambd
      self.optimizer = optimizer
      self.learning_rate = learning_rate
      self.beta1 = beta1
      self.beta2 = beta2
      self.epsilon = epsilon

      # Set model results as None
      self.costs = None
      self.train_accuracies = None
      self.val_accuracies = None

      # Set parameters as None
      self.parameters = None

  def add_input_layer(self, data):
      """
      This function adds the number of features as input layer to the layer list.

      Args:
      - data (array): the input data.

      Returns:
      - layers (list): layer dimensions including input layer.
      """
      # Check if the input data has two dimensions
      if len(data.shape) == 2:
        h, w = data.shape
        nb_features = h*w

      # Check if the input data has three dimensions
      elif len(data.shape) == 3:
        _, h, w = data.shape
        nb_features = h*w

      # Check if the input data has four dimensions
      elif len(data.shape) == 4:
        _, h, w, c = data.shape
        nb_features = h*w*c
      
      # Create a copy of the neural network's layer list 
      # and adds the number of features as the first element of the list.
      layers = self.layers.copy()
      layers.insert(0, nb_features)

      return layers
      
  def predict(self, data, flatten_data=True):
      """
      This function predicts the labels for the input data.

      Args:
      - data (array): input data for which to predict labels.
      - flatten_data (bool, optional): whether to flatten the data. Default to True.

      Returns:
      - y_pred (array): the predicted labels for the input data
      """
      L = len(self.parameters) // 2

      # Flatten data to predict on new data, not validation data
      if flatten_data:
        data = data.reshape(data.shape[0], -1).T         
        
      # Forward Propagation      
      A = data
      for l in range(1, L):
        A_prev = A
        Z = np.dot(self.parameters["W"+str(l)], A_prev) + self.parameters["b"+str(l)]
        A = activations.forward(Z, self.hidden_layer_activation)

      A_prev = A
      ZL = np.dot(self.parameters["W"+str(L)], A_prev) + self.parameters["b"+str(L)]
      AL = activations.forward(ZL, self.output_layer_activation)

      # Get predictions
      y_pred = AL

      return y_pred

  def train(self, train_data, train_label, epochs=10, batch_size=32, validation_split=0., drop_remainder=False):
      """
      This function trains the neural network model.

      Args:
      - train_data (array): the training data.
      - train_label (array): the training labels.
      - epochs (int, optional): the number of training epochs. Defaults to 10.      
      - batch_size (int, optional): the batch size for dividing the dataset into chunks. Defaults to 32.
      - validation_split (float, optional): the fraction of the training data to use for validation. Defaults to 0.
      - drop_remainder (bool, optional): whether to drop the last batch if it is smaller than the specified batch size. Defaults to False.

      Returns:
      - None
      """
      # Sets the random seed for reproducibility.
      np.random.seed(1)

      # Make empty the results just in case the train method run several times.
      self.costs = []
      self.train_accuracies = []
      self.val_accuracies = []

      # Add the number of features as input layer and the number of classes as output layer to the layer list
      layers = self.add_input_layer(train_data)
      output_layer = train_label.shape[-1]
      layers.append(output_layer)

      # Load and preprocess the datasets
      if validation_split == 0:
          train_minibatches = preprocess(train_data, train_label, batch_size, drop_remainder)
          len_train = len(train_minibatches)
      elif validation_split > 0:
          train_data, train_label, val_data, val_label = train_val_split(train_data, train_label, validation_split)
          train_minibatches = preprocess(train_data, train_label, batch_size, drop_remainder)
          val_minibatches = preprocess(val_data, val_label, batch_size, drop_remainder)
          len_train = len(train_minibatches)
          len_val = len(val_minibatches)
  
      # Set parameters and optimizer
      self.parameters = initializer.set_parameters(layers, self.kernel_initializer)
      if self.optimizer=="gd":
         pass
      elif self.optimizer=="momentum":
         v = initializer.set_momentum(self.parameters)
      elif self.optimizer=="rmsprop":
         s = initializer.set_rmsprop(self.parameters)
      elif self.optimizer=="adam":
         v, s, t = initializer.set_adam(self.parameters)
  
      # Build neural network
      nn = NetworkPropagation(self.hidden_layer_activation, self.output_layer_activation, self.dropout_rate)

      # Epoch loop
      for epoch in range(epochs):
      
          cost = 0.
          train_accuracy = 0.
          val_accuracy = 0
  
          # Train the model on train dataset.       
          for (X_train, y_train) in train_minibatches:
              # Forward propagation
              y_pred, caches = nn.forward(X_train, self.parameters)
              # Compute cost according to given loss type
              cost += compute_cost(y_train, y_pred, self.parameters, self.lambd, self.loss_type)
              # Backward propagation
              gradients = nn.backward(y_train, y_pred, self.parameters, caches, self.lambd)

              # Update gradients according to the optimizer
              updater = UpdateGradient(self.parameters, self.learning_rate, gradients)
              if self.optimizer == "gd":
                self.parameters = updater.gd()
              elif self.optimizer=="momentum":
                self.parameters, v = updater.momentum(v, self.beta1)
              elif self.optimizer=="rmsprop":
                self.parameters, s = updater.rmsprop(s, self.beta2, self.epsilon)
              elif self.optimizer=="adam":
                t = t + 1
                self.parameters, v, s = updater.adam(v, s, t, self.beta1, self.beta2, self.epsilon)
              
              # Calculate the accuracy score
              train_accuracy += accuracy_score(y_train, y_pred, self.output_layer_activation)
  
          # Get averages of evaluating metrics
          cost = np.round(cost / len_train, 4)
          train_accuracy = np.round(train_accuracy / len_train, 4)
          # Store the results
          self.costs.append(cost)
          self.train_accuracies.append(train_accuracy)
  
          if validation_split==0:
            print(f"Epoch {epoch+1} ---> Cost: {cost}\t Train Accuracy: {train_accuracy}")
  
          # Validate the model performance on validation dataset if split rate is bigger than 0.
          elif validation_split>0:
            for (X_val, y_val) in val_minibatches:
                # Predict the labels without flattening
                y_pred = self.predict(X_val, flatten_data=False)
                # Calculate the accuracy score
                val_accuracy += accuracy_score(y_val, y_pred, self.output_layer_activation)
            val_accuracy = np.round(val_accuracy / len_val, 4)
            # Store the results
            self.val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch+1} ---> Cost: {cost}\t Train Accuracy: {train_accuracy}\t Validation Accuracy: {val_accuracy}")
