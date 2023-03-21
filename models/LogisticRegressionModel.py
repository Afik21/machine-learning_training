
import numpy as np

# Now, we are going to create our model (mathematical function)
class LogisticRegressionModel(ModelSkeleton):
  def __init__(self,  X, Y, test_size = 0.2, iteration = 100, learning_rate = 0.01, random_state = True, validation_size = None):
    super().__init__(X, Y, test_size, random_state, validation_size)
    self.iteration = iteration
    self.learning_rate = learning_rate
    self.weigths = None
    self.bias = None

  def model(self, X):
    # sigmoid's function
    return 1/(1 + np.exp(-(np.dot(X, self.weigths) + self.bias)))

  def fit(self, X, y):
    # n_samples, n_features = X[:, np.newaxis].shape
    print(X.shape)
    n_samples, n_features = X.shape
    self.weigths = np.zeros(n_features)
    self.bias = 0

    for _ in range(self.iteration):
      y_pred = self.model(X)

      dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
      db = (1/n_samples) * np.sum(y_pred - y)

      self.weigths = self.weigths - self.learning_rate * dw
      self.bias = self.bias - self.learning_rate * db


  def loss(self, y_true, y_predicted):
    return np.mean((y_true - y_predicted)**2)

  def predict(self, X):
    y_pred = self.model(X)
    class_predictions = [1 if y > 0.5 else 0 for y in y_pred]
    return class_predictions

  def evaluate(self):
    pass

  def accuracy(self, y_test, y_pred):
    return np.sum(y_test==y_pred)/len(y_test)

  def plotLine(self, X, X_train, y_train, X_test, y_test, y_pred_line):
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8,6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X, y_pred_line, color='red', linewidth=2, label="Prediction")
    plt.show()