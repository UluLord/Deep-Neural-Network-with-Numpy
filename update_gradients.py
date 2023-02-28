import numpy as np

class UpdateGradient:
  """
  This class updates the model's parameters using different optimization algorithms.
  """
  def __init__(self, parameters, learning_rate, gradients):
      """
      Initializes the UpdateGradient class.

      Args:
      - params (dict): the parameters of the model.
      - learning_rate (float): the learning rate to use for the optimization algorithm.
      - gradients (dict): the gradients of the parameters computed during backpropagation.
      """
      self.layer_num = len(parameters)//2
      self.parameters = parameters
      self.learning_rate = learning_rate
      self.gradients = gradients
  
  def gd(self):
      """
      This function updates the model parameters using gradient descent optimization algorithm.

      Returns:
      - parameters (dict): the updated parameters of the model.
      """
      parameters = self.parameters
      
      for l in range(1, self.layer_num+1):

          parameters["W"+str(l)] = self.parameters["W"+str(l)] - self.learning_rate*self.gradients["dW"+str(l)]
          parameters["b"+str(l)] = self.parameters["b"+str(l)] - self.learning_rate*self.gradients["db"+str(l)]

      return parameters

  def momentum(self, v, beta1):
      """
      This function updates the model parameters using momentum optimization algorithm.
  
      Args:
      - v (dict): the exponentially weighted average of the gradients.
      - beta1 (float): the hyperparameter for the exponentially weighted average.
  
      Returns:
      - parameters (dict): the updated parameters of the model.
      - v (dict): the updated exponentially weighted average of the gradients.
      """
      parameters = self.parameters
      
      for l in range(1, self.layer_num+1):
      
          v["dW"+str(l)] = beta1*v["dW"+str(l)] + (1-beta1)*self.gradients["dW"+str(l)]
          v["db"+str(l)] = beta1*v["db"+str(l)] + (1-beta1)*self.gradients["db"+str(l)]

          parameters["W"+str(l)] = parameters["W"+str(l)] - self.learning_rate*v["dW"+str(l)]
          parameters["b"+str(l)] = parameters["b"+str(l)] - self.learning_rate*v["db"+str(l)]
  
      return parameters, v

  def rmsprop(self, s, beta2, epsilon):
      """
      This function updates the model parameters using RMSprop optimization algorithm.

      Args:
      - s (dict): the exponentially weighted average of the squared gradients for each parameter.
      - beta2 (float): the hyperparameter for the exponential decay of the past squared gradients.
      - epsilon (float): a small value added for numerical stability.

      Returns:
      - parameters (dict): the updated parameters after applying RMSprop.
      - s (dict): the updated exponentially weighted average of the squared gradients for each parameter.
      """
      parameters = self.parameters

      for l in range(1, self.layer_num+1):

          s["dW"+str(l)] = beta2*s["dW"+str(l)] + (1-beta2)*np.power(self.gradients["dW"+str(l)], 2)
          s["db"+str(l)] = beta2*s["db"+str(l)] + (1-beta2)*np.power(self.gradients["db"+str(l)], 2)

          parameters["W"+str(l)] = parameters["W"+str(l)] - self.learning_rate*(self.gradients["dW"+str(l)]/(np.sqrt(s["dW"+str(l)]) + epsilon))
          parameters["b"+str(l)] = parameters["b"+str(l)] - self.learning_rate*(self.gradients["db"+str(l)]/(np.sqrt(s["db"+str(l)]) + epsilon))

      return parameters, s

  def adam(self, v, s, t, beta1, beta2, epsilon):
      """
      This function updates the model parameters using Adam optimization algorithm.

      Args:
      - v (dict): the exponentially weighted average of the gradients for each parameter.
      - s (dict):  the exponentially weighted average of the squared gradients for each parameter.
      - t (dict): the current iteration number.
      - beta1 (float): the exponential decay rate for the first moment estimates.
      - beta2 (float): the exponential decay rate for the second moment estimates.
      - epsilon (float): a small value to avoid dividing by zero.

      Returns:
      - parameters (dict): the updated parameters.
      - v (dict): the updated exponentially weighted average of the gradients for each parameter.
      - s (dict): the updated exponentially weighted average of the squared gradients for each parameter.
      """
      parameters = self.parameters
      v_corrected = {}
      s_corrected = {}

      for l in range(1, self.layer_num+1):

          # Update biased first moment estimate
          v["dW"+str(l)] = beta1*v["dW"+str(l)] + (1-beta1)*self.gradients["dW"+str(l)]
          v["db"+str(l)] = beta1*v["db"+str(l)] + (1-beta1)*self.gradients["db"+str(l)]
          # Correct bias in first moment estimate
          v_corrected["dW"+str(l)] = v["dW"+str(l)]/(1-np.power(beta1, t))
          v_corrected["db"+str(l)] = v["db"+str(l)]/(1-np.power(beta1, t))

          # Update biased second moment estimate
          s["dW"+str(l)] = beta2*s["dW"+str(l)] + (1-beta2)*np.power(self.gradients["dW"+str(l)], 2)
          s["db"+str(l)] = beta2*s["db"+str(l)] + (1-beta2)*np.power(self.gradients["db"+str(l)], 2)
          # Correct bias in second moment estimate
          s_corrected["dW"+str(l)] = s["dW"+str(l)]/(1-np.power(beta2, t))
          s_corrected["db"+str(l)] = s["db"+str(l)]/(1-np.power(beta2, t))

          parameters["W"+str(l)] = parameters["W"+str(l)] - self.learning_rate*(v_corrected["dW"+str(l)]/(np.sqrt(s_corrected["dW"+str(l)]) + epsilon))
          parameters["b"+str(l)] = parameters["b"+str(l)] - self.learning_rate*(v_corrected["db"+str(l)]/(np.sqrt(s_corrected["db"+str(l)]) + epsilon))

      return parameters, v, s