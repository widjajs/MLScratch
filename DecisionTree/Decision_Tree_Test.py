import numpy as np
from sklearn.model_selection import train_test_split
from Decision_Tree import DecisionTree
import pandas as pd

col_names = ['poisonous', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor',
             'gill_attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
             'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
             'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
             'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

df = pd.read_csv('mushroom.csv', names=col_names)

def accuracy(y_actual, y_pred):
    return np.sum(y_actual == y_pred) / len(y_actual)

X = df.drop("poisonous", axis=1).to_numpy()
y = df['poisonous'].map({'p': 1, 'e': 0}).to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=1234)

tree = DecisionTree(max_depth=6)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print("Accuracy: ", accuracy(y_test, y_pred))

