import numpy as np
import activations

class NetworkPropagation:
    """
    This class performs forward propagation in a neural network and backward propagation in the neural network.
    """
    def __init__(self, hidden_layer_activation, output_layer_activation, dropout_rate):
        """
        Initializes the Propagation object.

        Args:
        - hidden_layer_activation (str): activation function to use in the hidden layers.
        - output_layer_activation (str): activation function to use in the output layer.
        - dropout_rate (float): dropout rate to close random units during training.
        """
        self.hidden_layer_activation = hidden_layer_activation
        self.output_layer_activation = output_layer_activation
        self.keep_prob = 1 - dropout_rate

    def forward(self, data, parameters):
        """
        This function performs forward propagation in the neural network.

        Args:
        - data (array): input data.
        - parameters (dict): the weights and biases of the neural network.

        Returns:
        - AL (array): the output of the neural network 
        - caches (dict): the results in the network.
        """
        L = len(parameters) // 2
        caches = {}

        # Input layer
        A = data
        caches["A0"] = A

        # Loop for hidden layers
        for l in range(1, L):
            # Previous layer output
            A_prev = A
            # Multiply weights with previous layer output and add to biases
            Z = np.dot(parameters["W"+str(l)], A_prev) + parameters["b"+str(l)]
            # Activation function
            A = activations.forward(Z, self.hidden_layer_activation)
            # Dropout process
            D = np.random.rand(A.shape[0], A.shape[1]) < self.keep_prob
            A = (A*D) / self.keep_prob
            # Store the results to caches
            caches["Z"+str(l)] = Z
            caches["A"+str(l)] = A
            caches["D"+str(l)] = D

        # For output layer
        A_prev = A
        ZL = np.dot(parameters["W"+str(L)], A_prev) + parameters["b"+str(L)]
        AL = activations.forward(ZL, self.output_layer_activation)
        caches["Z"+str(L)] = ZL
        caches["A"+str(L)] = AL

        return AL, caches
    

    def backward(self, real_label, pred_label, parameters, caches, lambd):
        """
        This function computes gradients of the cost function with respect to the parameters of the model using backpropagation.

        Args:
        - real_label (array): the true labels of the data.
        - pred_label (array): the predicted labels of the data.
        - parameters (dict): the weights and biases of the neural network.
        - caches (dict): the intermediate values needed for backpropagation.
        - lambd (float): L2 regularization parameter.

        Returns:
        - gradients (dict): the gradients of the cost function with respect to the parameters of the model.
        """
        L = len(parameters) // 2
        len_data = real_label.shape[1]
        gradients = {}

        # For last layer
        # Derivatives for last layer
        dZL = pred_label - real_label
        # Derivatives for last layer weights
        dWL = (1/len_data)*np.dot(dZL, caches["A"+str(L-1)].T) + (1/len_data)*(parameters["W"+str(L)]*lambd)
        # Derivatives for last layer biases
        dbL = (1/len_data)*np.sum(dZL, axis=1, keepdims=True)
        # Derivatives for last layer activation with dropout
        dA_prev = np.dot(parameters["W"+str(L)].T, dZL)
        dA_prev = np.multiply(dA_prev, caches["D"+str(L-1)])/self.keep_prob
        # Store the derivatives
        gradients["dW"+str(L)] = dWL
        gradients["db"+str(L)] = dbL

        # For hidden layers
        for l in reversed(range(2, L)):
            dA = dA_prev
            # Derivatives for current layer
            dZ = np.multiply(dA, activations.backward(caches["A"+str(l)], self.hidden_layer_activation))
            # Derivatives for current layer weights
            dW = (1/len_data)*np.dot(dZ, caches["A"+str(l-1)].T) + (1/len_data)*(parameters["W"+str(l)]*lambd)
            # Derivatives for current layer biases
            db = (1/len_data)*np.sum(dZ, axis=1, keepdims=True)
            # Derivatives for current layer activation with dropout
            dA_prev = np.dot(parameters["W"+str(l)].T, dZ)
            dA_prev = (dA_prev*caches["D"+str(l-1)])/self.keep_prob
            # Store the derivatives
            gradients["dW"+str(l)] = dW
            gradients["db"+str(l)] = db
        
        # For first layer
        dA = dA_prev
        # Derivatives for first layer
        dZ1 = np.multiply(dA, activations.backward(caches["A1"], self.output_layer_activation))
        # Derivatives for first layer weights
        dW1 = (1/len_data)*np.dot(dZ1, caches["A0"].T) + (1/len_data)*(parameters["W1"]*lambd)
        # Derivatives for first layer biases
        db1 = (1/len_data)*np.sum(dZ1, axis=1, keepdims=True)
        # Store the derivatives
        gradients["dW1"] = dW1
        gradients["db1"] = db1

        return gradients