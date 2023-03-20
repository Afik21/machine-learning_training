import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Now, we are going to create our model (mathematical function)

from ModelSkeleton import ModelSkeleton

class LinearModel(ModelSkeleton):
  def __init__(self,  X, Y, iteration = 100, learning_rate = 0.01):
    super().__init__(self, X, Y)
    self.X = X
    self.Y = Y
    self.iteration = iteration
    self.learning_rate = learning_rate
    self.weigths = None
    self.bias = None

  def model(self, X):
    return np.dot(X, self.weigths) + self.bias

  def fit(self, X, y):
    n_samples, n_features = X.shape
    self.weigths = np.zeros(n_features)
    self.bias = 0

    for _ in range(self.iteration):
      y_pred = self.model(X)

      dw = (1/n_samples) * np.dot(X, (y_pred - y))
      db = (1/n_samples) * np.sum(y_pred - y)

      self.weigths = self.weigths - self.learning_rate * dw
      self.bias = self.bias - self.learning_rate * db


  def loss(y_test, predictions):
    return np.mean((y_test - predictions)**2)

  def predict(self, X):
    y_pred = np.dot(X, self.weigths) + self.bias
    return y_pred

  def evaluate(self):
    pass

  def accuracy(self):
    pass