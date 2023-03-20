class ModelSkeleton:
  def __init__(self, X, Y, test_size = 0.2, random_state=True, validation_size = None):
    self.X = X
    self.Y = Y
    self.test_size = test_size
    self.random_state = random_state
    self.validation_size = validation_size

  def split(self):
    if self.random_state:
      np.random.seed(0)

    # Shuffling Data before split
    indices = np.arange(len(self.X))
    np.random.shuffle(indices)

    # Now, we are going to split
    size = int(len(X) - (len(X) * self.test_size))
    train_idx = indices[: size]
    test_idx = indices[size:]
    X_train, y_train = self.X[train_idx], self.Y[train_idx]
    X_test, y_test = self.X[test_idx], self.Y[test_idx]

    return X_train, y_train, X_test, y_test

  def fit(self):
    pass

  def predict(self):
    pass

  def evaluate(self):
    pass

  def accuracy(self):
    pass