# Deep Neural Network with Numpy

A Deep Neural Network (DNN) is a type of artificial neural network (ANN) that is designed to learn from complex and large datasets. It is composed of multiple layers of interconnected nodes or neurons that are capable of processing and analyzing input data. DNN models are used for various machine learning tasks such as image recognition, speech recognition, natural language processing, and predictive analytics.


The architecture of a DNN model consists of an input layer, one or more hidden layers, and an output layer. Each layer is composed of multiple neurons that are interconnected with the neurons in the adjacent layers. The input layer receives the raw data, which is then processed by the hidden layers through a series of mathematical computations, and finally, the output layer produces the desired output.

![neural](https://user-images.githubusercontent.com/99184963/221975755-61b750a8-5fd4-4b3a-b6c4-7613d4b42c9c.jpg)

> Image Credit: Mathworks

DNN models use a technique called backpropagation to train the network. During the training process, the weights of the connections between the neurons are adjusted to minimize the error between the predicted output and the actual output. The training process is typically performed using a large dataset, and the network continues to adjust its weights until it achieves a desired level of accuracy.


This repository contains an implementation of a Deep Neural Network (DNN) using only the NumPy library in Python.

## Requirement

This code was developed and tested using NumPy 1.24.1. It may work with other versions of NumPy, but this has not been tested.

## Usage

You can train your datasets (like .csv files, images) in this DNN Model.
To use the DNN Model for your projects, clone this repository like;

     git clone https://github.com/UluLord/Deep-Neural-Network-with-Numpy.git

After cloning, follow the instructions below;

* Change your directory to this repository path.

> Example usage;

     import os
     os.chdir("./Deep-Neural-Network-with-Numpy")

* Import **DNNModel** class from the **model.py** file to your enviroment. 

> Example usage;

     from model import DNNModel

* After that, create an instance of the DNNModel with desired arguments;
  
  ◦ **layers:** The number of neurons in each hidden layer. Neurons should be specified in a list.
  
  ◦ **hidden_layer_activation:** The activation function used in hidden layers. Default is ‘relu’.
  
  ◦ **output_layer_activation:** The activation function used in output layer. Default is ‘softmax’.
  
  ◦ **kernel_initializer:** Method to initialize the kernel. Default is ‘he’.
  
  ◦ **loss_type:** Loss function to compute the model cost. Default is ‘categorical_crossentropy’.
  
  ◦ **dropout_rate:** The dropout rate closing some neurons randomly in the hidden layers during training. Default is 0.
  
  ◦ **lambd:** The L2 regularization penalty used during training. Defaults to 0.
  
  ◦ **optimizer:** Model optimizer. Default is 'adam'.
  
  ◦ **learning_rate:** The learning rate used during training. Defaults to 0.001
  
  ◦ **beta1:** The hyperparameter for Momentum or Adam optimizers. Default is 0.9.
  
  ◦ **beta2:** The hyperparameter for RMSprop or Adam optimizers. Default is 0.999.
  
  ◦ **epsilon:** A small constant added to the denominator to prevent division by zero in RMSProp and Adam optimizers. Default is 1e-8.


* Then, use the **‘train’** method with desired arguments to train the network on your dataset;
  
  ◦ **train_data:** The training data.
  
  ◦ **train_label:** The training labels.
  
  ◦ **epochs:** The number of training epochs. Defaults is10.
  
  ◦ **batch_size:** The batch size for dividing the dataset into chunks. Defaults is 32.
  
  ◦ **validation_split:** The fraction of the training data to use for validation. Defaults is 0.
  
  ◦ **drop_remainder:** Whether to drop the last batch if it is smaller than the specified batch size. Defaults is False.

> Example usage;

     dnn_model = DNNModel(layers=[32, 64], hidden_layer_activation="relu", output_layer_activation="softmax", 
                          kernel_initializer="he", loss_type="categorical_crossentropy", dropout_rate=0.3, lambd=0.1, 
                          optimizer="adam", learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8)

     dnn_model.train(train_data, train_label, 
                     epochs=10, 
                     batch_size=32, 
                     validation_split=0.3, 
                     drop_remainder=False)


> To reach the cost results of the trained model which are stored every epoch in a list;

    
     dnn_model.costs


> To reach the train accuracy scores of the trained model which are stored every epoch in a list;

     dnn_model.train_accuracies


> To reach the validation accuracy scores of the trained model which are stored every epoch in a list;


     dnn_model.val_accuracies


  **NOTE**: if the validation split is equal to 0, the output will be an empty list.

* To predict new data with the trained model, use the ‘predict’ method with desired arguments;
  
  ◦ data: Input data to be used for label prediction.
  
  ◦ flatten_data: Whether to flatten the input data. ‘True’ is mandatory to predict labels of new data. Default is True.

> Example usage;

     predict_labels = dnn_model.predict(data, flatten_data=True)


## Citation

If you use this repository in your work, please consider citing us as the following.

    @misc{ululord2023deep-neural-network-with-numpy,
          author = {Fatih Demir},
          title = {Deep Neural Network with Numpy},
          date = {2023-03-01},
          url = {https://github.com/UluLord/Deep-Neural-Network-with-Numpy}
          }
