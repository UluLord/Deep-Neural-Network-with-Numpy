import numpy as np

def forward(Z, activation):
    """
    This function calculates the activation of a layer based on the input Z and the activation function.

    Args:
    - Z (array): the input to the layer.
    - activation (str): the activation function to be used.

    Returns:
    - A (array): the activation of the layer.
    """

    if activation == "sigmoid":
        A = 1 / (1 + np.exp(-Z))
        
    elif activation == "tanh":
        A = np.tanh(Z)

    elif activation == "leaky_relu":
        A = np.maximum(0.01*Z, Z)

    elif activation == "relu":
        A = np.maximum(0, Z)

    elif activation == "softmax":
        exp = np.exp(Z)
        sum = np.sum(exp, axis=1, keepdims=True)
        A = exp / sum

    return A


def backward(A, activation):
    """
    This function calculates the derivative of the activation of a layer based on the input A and the activation function.

    Args:
    - A (array): the activation of the layer.
    - activation (str): the activation function to be used.

    Returns:
    - dA (array): the derivative of the activation of the layer.
    """

    if activation == "sigmoid" or activation == "softmax":
        dA = A * (1 - A)

    elif activation == "tanh":
        dA = 1 - np.power(A, 2)

    elif activation == "leaky_relu":
        dA = np.ones_like(A)
        dA[A <= 0] = 0.01

    elif activation == "relu":
        dA = np.int64(A > 0)

    return dA