import numpy as np
from collections import Counter
# how K nearest number works given a data point
    # find the distance from point to all other data points
    # get the closest K points
    # with Regression -> get average of values
    # with Classification -> majority vote (majority of k points are closer)
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2) # given v = x1x2 find distance

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictons = [self._predict(x) for x in X]
        return predictons

    def _predict(self, x):
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get k nearest samples, labels
        k_indices = np.argsort(distances)[0:self.k] # sorts distances but with indices
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # get most common class label
        most_common = Counter(k_nearest_labels).most_common(1) # returns tuple -> (label, # of appearances)
        return most_common[0][0]




