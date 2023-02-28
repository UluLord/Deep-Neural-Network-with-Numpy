import numpy as np

def compute_cost(real_label, pred_label, parameters, lambd, loss_type):
    """
    This function calculates the binary or categorical cross-entropy loss between the predicted values pred_label and the true values real_label.

    Args:
    - real_label (array): the true output values.
    - pred_label (array): the predicted output values.
    - parameters (dict): the weights and biases of the neural network.
    - lambd (float): L2 regularization parameter.
    - loss_type (str): the type of loss to compute, either 'binary_crossentropy' or 'categorical_crossentropy'.

    Returns:
    - cost (float): the binary or categorical cross-entropy loss.
    """
    L = len(parameters) // 2
    len_data = real_label.shape[1]

    # Compute cross entropy cost
    crossentropy_cost = None
    if loss_type == "binary_crossentropy":
        crossentropy_cost = (-1/len_data)*np.sum(np.multiply(real_label, np.log(pred_label)) + np.multiply(1-real_label, np.log(1-pred_label)))
    elif loss_type == "categorical_crossentropy":
        crossentropy_cost = (-1/len_data)*np.sum(np.multiply(real_label, np.log(pred_label)))

    # L2 regularization
    sum_squared_W = 0
    for l in range(1, L+1):
        cur_W = parameters["W" + str(l)]
        sum_squared_W = sum_squared_W + np.sum(np.square(cur_W))
    L2_reg = (sum_squared_W*lambd)/(2*len_data)

    # Get final cost
    cost = crossentropy_cost + L2_reg
    cost = np.squeeze(np.array(cost))
    return cost

def accuracy_score(y_true, y_pred, output_layer_activation):
    """
    This function calculates the accuracy score of the predicted labels with respect to the true labels.

    Args:
    - y_true (array): the true labels.
    - y_pred (array): the predicted labels.

    Returns:
    - score (float): the accuracy score.
    """
    # Determine the true and predicted labels based on the output layer activation
    if output_layer_activation=="sigmoid":
       true_labels = y_true
       pred_labels = np.round(y_pred)

    elif output_layer_activation=="softmax":
       true_labels = np.argmax(y_true, axis=0)
       pred_labels = np.argmax(y_pred, axis=0)

    # Calculate the accuracy score as the ratio of correctly classified samples to the total number of samples
    score = np.sum(true_labels == pred_labels) / len(true_labels)
    return score