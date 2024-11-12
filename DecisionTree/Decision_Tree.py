import numpy as np
from collections import Counter

# calculates entropy: -sum(p(X) * log2(p(X))) where p(X) = #x/n
def entropy(y):
    hist = np.bincount(y)
    p_list = hist/len(y)
    return -np.sum([p * np.log2(p) for p in p_list if p > 0])

# node helper class
class Node:
    def __init__(self, split_feat=None, split_thresh=None, left=None, right=None, *, leaf_val=None ):
        self.split_feat = split_feat
        self.split_thresh = split_thresh
        self.left = left
        self.right = right
        self.leaf_val = leaf_val

    def is_leaf(self):
        return self.leaf_val is not None

class DecisionTree:
    def __init__(self, min_samples=2, max_depth=100, n_feats=None):
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    # grow tree X = train data, y = labels
    def fit(self, X, y):
        if not self.n_feats: # sets n_feats to num features in X
            self.n_feats = X.shape[1]
        else: # makes sure n_feats !greater than X num features
            self.n_feats = min(self.n_feats, X.shape[1])

        self.root = self._grow_tree(X, y)

    # grow tree helper
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stop conditions
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples):
            leaf_value = self._most_common_label(y)
            return Node(leaf_val=leaf_value)

        # selects random subset of feature indices
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedy search
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # split tree with best feat and thresh
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        # call grow recursively and continue same process until leaf node found
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)

    # get most common label out of list
    def _most_common_label(self, y):
        counter = Counter(y)
        # counter.most_common(1) returns most common labels in list of tuples in descending order
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        # go through all the features and get the one with the best information gain
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)
            for threshold in thresholds:
                gain = self._info_gain(y, X_col, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    # calculates info gain: Entropy(parent) - [weighted avg] * Entropy(child)
    def _info_gain(self, y, X_col, threshold):
        # Entropy parent
        parent_entropy = entropy(y)

        # make a split
        left_idxs, right_idxs = self._split(X_col, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            # if left or right child 0 then no info gained
            return 0

        # weighted avg of children * Entropy(child)
        n_samples = len(y)
        n_left_samples, n_right_samples = len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_left_samples/n_samples) * entropy_left + (n_right_samples/n_samples) * entropy_right

        return parent_entropy - child_entropy


    # create left and right split helper function
    def _split(self, X_col, threshold):
        # split left for mushroom traits that do NOT match threshold
        left_idxs = np.argwhere(X_col != threshold).flatten()
        # split right for mushroom traits that DO match threshold
        right_idxs = np.argwhere(X_col == threshold).flatten()
        return left_idxs, right_idxs


    # traverse tree
    def predict(self, X):
        # predict lethality
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.leaf_val

        if x[node.split_feat] != node.split_thresh:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)
