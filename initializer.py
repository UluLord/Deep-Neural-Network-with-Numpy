import numpy as np

def set_parameters(layers, kernel_initializer):
    """
    This function initializes the weights and biases of the neural network using the specified kernel initializer.

    Args:
    - layers (list): the number of units in each layer of the neural network.
    - kernel_initializer (str): the type of kernel initializer to use for the neural network weights.

    Returns:
    - parameters (dict): the weights and biases of the neural network.
    """
    L = len(layers)
    parameters = {}

    # Set the weights and biases of the neural network according to kernel initializer type
    for l in range(1, L):

        if kernel_initializer == "zeros":
           parameters["W"+str(l)] = np.zeros((layers[l], layers[l-1]))
           parameters["b"+str(l)] = np.zeros((layers[l], 1))

        elif kernel_initializer == "random":
           parameters["W"+str(l)] = np.random.randn(layers[l], layers[l-1])*0.01
           parameters["b"+str(l)] = np.zeros((layers[l], 1))

        elif kernel_initializer == "xavier":
           parameters["W"+str(l)] = np.random.randn(layers[l], layers[l-1])*np.sqrt(1/layers[l-1])
           parameters["b"+str(l)] = np.zeros((layers[l], 1))

        elif kernel_initializer == "he":
           parameters["W"+str(l)] = np.random.randn(layers[l], layers[l-1])*np.sqrt(2/layers[l-1])
           parameters["b"+str(l)] = np.zeros((layers[l], 1))

    return parameters

def set_momentum(parameters):
    """
    This function initializes the velocity for the momentum optimizer.
    
    Args:
    - parameters (dict): the weights and biases of the neural network.
    
    Returns:
    - v (dict): the velocity for each weight and bias parameter.
    """
    L = len(parameters) // 2
    v = {}
    
    # Set the velocity for each weight and bias parameter.
    for l in range(1, L+1):

        v["dW"+str(l)] = np.zeros_like(parameters["W"+str(l)])
        v["db"+str(l)] = np.zeros_like(parameters["b"+str(l)])

    return v

def set_rmsprop(parameters):
    """
    This function initializes the exponential moving average for the RMSprop optimizer.
    
    Args:
    - parameters (dict): the weights and biases of the neural network.
    
    Returns:
    - s (dict): the exponential moving average for each weight and bias parameter.
    """
    L = len(parameters) // 2
    s = {}

    # Set the exponential moving average for each weight and bias parameter
    for l in range(1, L+1):
      s["dW"+str(l)] = np.zeros_like(parameters["W"+str(l)])
      s["db"+str(l)] = np.zeros_like(parameters["b"+str(l)])

    return s
  
def set_adam(parameters):
    """
    This function performs the Adam optimization algorithm.

    Args:
    - parameters (dict): the weights and biases of the neural network.

    Returns:
    - v (dict): the velocity for each weight and bias parameter.
    - s (dict): the exponential moving average for each weight and bias parameter.
    - t (int): the current iteration number.
    """
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(1, L+1):
      # Set first moment estimate
      v["dW"+str(l)] = np.zeros_like(parameters["W"+str(l)])
      v["db"+str(l)] = np.zeros_like(parameters["b"+str(l)])
      
      # Set second moment estimate
      s["dW"+str(l)] = np.zeros_like(parameters["W"+str(l)])
      s["db"+str(l)] = np.zeros_like(parameters["b"+str(l)])

    # Set current iteration
    t = 0

    return v, s, t