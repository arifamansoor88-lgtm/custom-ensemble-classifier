####################################
# Random Forest
####################################
import numpy as np
import pandas as pd
import copy
from typing import Tuple, Any, Optional, Union, List
from collections import Counter
import random
import math
import pickle
import re

class DecisionTreeClassifier:
    """ My implementation of sklearn's Decision Tree Classifier.

    Note: this assumes that ALL features are discrete i.e. categorical.

    SOURCES:
    - https://medium.com/@cristianleo120/master-decision-trees-and-building-them-from-scratch-in-python-af173dafb836
    - https://www.kaggle.com/code/fareselmenshawii/decision-tree-from-scratch

    """
    def __init__(self, min_samples_split: int = 10, in_forest: bool = False, random_state: Optional[int] = None):
        """ Initialize the decision tree classifier.

        - min_samples_split: the minimum number of samples required to split an internal node.
          I left out max_depth because during initial testing, the best value for it was None.
        - in_forest: a boolean value which is True iff this tree is part of a forest
        - random_state: an integer for reproducibility
        - tree: the decision tree, as a nested dictionary...who is OOP idk her
        """
        self.min_samples_split = min_samples_split
        self.in_forest = in_forest
        self.tree = None
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray):
        """ Train the model by building the tree based on X & y.

        - X: training data matrix
        - y: associated labels
        Please ensure that both X & y are numpy arrays and not pandas dataframes
        or series.
        Assumes that all features are categorical, so make sure floats are rounded
        to ints.
        """
        self.tree = self.build_tree(X, y)

    def build_tree(self, X: np.ndarray, y: np.ndarray) -> dict:
        """ Train the model and return a dictionary representing the decision tree.

        - X: training data matrix
        - y: associated labels

        The dictionary can be either a leaf or a subtree node representing a split
        e.g.,
        - leaf node {'label': 'Shawarma'}
        - subtree node: {'feature': 'Avengers', 'value': 1, 'left': [data points['Avengers'] == 1] ,
          'right': [all other data points] }
        """
        # base case 1: all samples have same label
        if len(np.unique(y)) <= 1:
            return {'label': np.unique(y)[0]}
        # base case 2: min_samples_split reached
        if len(y) < self.min_samples_split:
            unique_vals, counts = np.unique(y, return_counts=True)
            majority_label = unique_vals[np.argmax(counts)]
            return {'label': majority_label}

        # recursive case
        best_split = self.find_best_split(X, y)
        if best_split is None:
            unique_vals, counts = np.unique(y, return_counts=True)
            majority_label = unique_vals[np.argmax(counts)]
            return {'label': majority_label}

        feature, split_on, left_X, left_y, right_X, right_y = best_split
        # build subtrees recursively
        left_subtree = self.build_tree(left_X, left_y)
        right_subtree = self.build_tree(right_X, right_y)

        return {'feature': feature,
                'value': split_on,
                'left': left_subtree,
                'right': right_subtree}

    def find_best_split(self, X: np.ndarray, y: np.ndarray) -> Optional[Tuple[str, Union[int, float, str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """ Use Gini impurity to find the find the best feature and value to split on using Gini impurity.

        - X: training data matrix
        - y: associated labels

        Returns a tuple containing the best feature, value, and its split:
        left_X, left_y, right_X, right_y, or None if no split is found.
        """
        best_gini = float('inf')
        best_split = None

        # if tree is part of forest, randomly choose subset of features
        # subset size is sqrt(n_features)
        n_features = X.shape[1]
        features = list(range(n_features))
        if self.in_forest:
            max_features = round(math.sqrt(n_features))
            if self.random_state:
                random_state = np.random.RandomState(self.random_state) # to avoid impacting global numpy state
                features = random_state.choice(features, max_features, replace=False).tolist()
            else:
                features = np.random.choice(features, max_features, replace=False).tolist()

        for feature in features:  # test each feature
            col = X[:, feature]
            vals = np.unique(col)
            for val in vals:  # test each split of this feature: one side == value, rest == not value
                mask_left = col == val  # masque for vectorrrrization
                mask_right = ~mask_left
                y_l = y[mask_left]
                y_r = y[mask_right]
                # i am paranoid that a split w min samples will escape to here
                if len(y_l) >= self.min_samples_split and len(y_r) >= self.min_samples_split:
                    # calc Gini impurity of split
                    split_impurity = self.split_impurity(y_l, y_r)
                    if split_impurity < best_gini:
                        best_gini = split_impurity
                        best_split = (feature, val, X[mask_left], y_l, X[mask_right], y_r)

        return best_split

    def gini(self, y: np.ndarray) -> float:
        """ Return the Gini impurity of a series y. Gini impurity = how often a
        random datapoint would be labelled incorrectly. """
        # Gini = 1 - sum(p^2)
        if len(y) == 0:
            return 0.0
        unique_vals, class_counts = np.unique(y, return_counts=True)
        probabilities = class_counts / len(y)
        impurity = 1 - np.sum(probabilities ** 2)
        return impurity

    def split_impurity(self, left_y: np.ndarray, right_y: np.ndarray) -> float:
        """ Return the Gini impurity for a split. Remember: a lower Gini score
        means the split is better.

        - left_y: labels for the left data points (== value) after the split.
        - right_y: labels for the right data points (!= value) after the split.
        """
        # gini impurity for a split = n_left / n_total * Gini_left + n_right / n_total * Gini_right
        Gini_l = self.gini(left_y)
        Gini_r = self.gini(right_y)
        n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right

        split_impurity = (n_left / n_total * Gini_l) + (n_right / n_total * Gini_r)
        return split_impurity

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Return the predicted class for each data point in X.
        Please ensure that X is a numpy array and not a pandas dataframe or series.
        Assumes that all features are categorical, so make sure floats are rounded
        to ints.
        """
        predictions = np.array([self.predict_single(row, self.tree) for row in X])
        return predictions

    def predict_single(self, row: np.ndarray, tree: dict) -> Any:
        """ For ONE test point, go down the decision tree and return the
        predicted class.

        - row: a single test data point
        - tree: the current decision tree/subtree/leaf we are at
        """
        # base case: there is a leaf in self.tree
        if 'label' in tree:
            return tree['label']

        # recursive case: there is a tree with branches in self.tree
        if row[tree['feature']] == tree['value']:  # left subtree
            return self.predict_single(row, tree['left'])
        else:  # right subtree
            return self.predict_single(row, tree['right'])


class RandomForestClassifier():
    """ My implementation of sklearn's RFC.
    Bootstrapping and categorical features are assumed. Uses sqrt for max_features.

    SOURCES:
    - https://medium.com/@enozeren/building-a-random-forest-model-from-scratch-81583cbaa7a9
    """
    def __init__(self, n_estimators: int = 200, max_samples: Optional[float] = None, min_samples_split: int = 10, random_state: Optional[int] = None) -> None:
        """ Initialize the RFC.

        - n_estimators: the number of individual decision trees in this forest
        - min_samples_split: the minimum number of samples needed to split an internal node
        - max_samples: a float in the range (0.0, 1.0] which specifies the size of the bootstrap batch
        used to train each individual tree, as a proportion of the original dataset size.
        - random_state: for reproducibility
        """
        if max_samples:
            if (max_samples <= 0.0) or (max_samples > 1.0):
                raise Exception("max_samples must be in the range (0.0, 1.0].")

        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_samples = max_samples
        self.n_classes = None
        self.trees = []
        self.random_state = random_state

    def get_bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Get one bootstrap sample the size of self.max_samples * N (where N is the original
        number of samples). Return this as a tuple of [X_sample, y_sample].

        - X: training data matrix
        - y: associated labels
        """
        if self.random_state:  # don't touch global numpy state
            random_state = np.random.RandomState(self.random_state)

        n = X.shape[0]  # no. samples
        if self.max_samples:  # max_samples is not None
            bstrap_size = max(round(n * self.max_samples), 1)
            if self.random_state:
                indices = random_state.choice(n, size=bstrap_size, replace=True)  # sample w replacement
            else:
                indices = np.random.choice(n, size=bstrap_size, replace=True)  # sample w replacement
        else:  # max_samples is None, sample full thing w/replacement (does not usually yield same dataset!)
            if self.random_state:
                indices = random_state.choice(n, size=n, replace=True)
            else:
                indices = np.random.choice(n, size=n, replace=True)

        X_boot = X[indices]
        y_boot = y[indices]
        return X_boot, y_boot

    def fit(self, X: np.ndarray, y: np.ndarray):
        """ Fit the RFC to the training data.

        - X: training data matrix
        - y: associated labels

        Please ensure that both X and y are numpy arrays and not pandas dataframes
        or series.
        Assumes that all features are categorical, so make sure floats are rounded
        to ints.
        """
        self.n_classes = len(np.unique(y))
        for i in range(self.n_estimators):
            X_boot, y_boot = self.get_bootstrap_sample(X, y)
            tree = DecisionTreeClassifier(min_samples_split=self.min_samples_split, in_forest=True, random_state=self.random_state)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Predict class labels for a set of samples.

        Please ensure that X is a numpy array and not a pandas dataframe or series.
        Assumes that all features are categorical, so make sure floats are rounded
        to ints.
        """
        tree_predictions = np.empty((self.n_estimators, X.shape[0]), dtype=object)

        # get predictions from each tree
        for i, tree in enumerate(self.trees):
            tree_predictions[i, :] = tree.predict(X)

        # take the mode (most common value) across trees for each sample
        majority_votes = []
        for i in range(X.shape[0]):
            sample_votes = tree_predictions[:, i]
            unique_values, counts = np.unique(sample_votes, return_counts=True)
            majority_vote = unique_values[np.argmax(counts)]
            majority_votes.append(majority_vote)

        return np.array(majority_votes)

    def save(self, filename: str):
        """ Save the RFC model to a pickle file with filename so it can be used
        out-of-the-box.
        filename must be a pickle (.pkl) file, e.g. "RFC_pretrained.pkl".
        """
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.trees, f)
            print(f"Success! RFC exported to {filename}.")
        except pickle.PicklingError:
            print("Error: RFC could not be pickled. Double-check the types of data in the tree :(")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def load_pretrained(self, filename: str):
        """ Load a pretrained RFC from filename.
        filename must be a pickle (.pkl) file which contains a list of dictionaries,
        where each dict represents one decision tree in the forest. """
        try:
            with open(filename, "rb") as f:
                self.trees = pickle.load(f)
            print(f"Success! Pre-trained RFC loaded from {filename}.")
        except pickle.UnpicklingError:
            print(f"Error: either {filename} is not a valid pickle file or is corrupted.")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def predict_proba(self, X: np.ndarray) -> dict:
        """
        Returns predicted probabilities as a dictionary: {class_label: probability_vector}
        Each vector contains the estimated probability of that class for each input row.
        """
        n_samples = X.shape[0]
        class_labels = ["Pizza", "Sushi", "Shawarma"]
        class_index = {label: i for i, label in enumerate(class_labels)}
        votes = np.zeros((n_samples, len(class_labels)))

        for tree in self.trees:
            preds = tree.predict(X)
            for i, pred in enumerate(preds):
                votes[i, class_index[pred]] += 1

        probs = votes / len(self.trees)
        return {label: probs[:, idx] for idx, label in enumerate(class_labels)}
