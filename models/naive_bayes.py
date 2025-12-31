####################################
# Naive Bayes
####################################
import numpy as np
import random
import pickle

class NaiveBayesClassifier():
    def __init__(self, *, a=2, b=2, split=90, N=100) -> None:
        self.a = a
        self.b = b
        self.split = split
        self.N = int(N)

    def _map_pi_theta(self, X, y):
        a = self.a
        b = self.b
        N, vocab_size = X.shape[0], X.shape[1]
        pi = 0
        theta = np.zeros([vocab_size, 3])

        X_pizza = X[y == "Pizza"]
        X_sushi = X[y == "Sushi"]
        X_shawarma = X[y == "Shawarma"]

        N_pizza = X_pizza.shape[0]
        N_sushi = X_sushi.shape[0]
        N_shawarma = X_shawarma.shape[0]

        theta[:, 0] = (np.matmul(np.transpose(X_pizza), np.ones(N_pizza)) + a - 1) / (N_pizza + a + b - 2)
        theta[:, 1] = (np.matmul(np.transpose(X_sushi), np.ones(N_sushi)) + a - 1) / (N_sushi + a + b - 2)
        theta[:, 2] = (np.matmul(np.transpose(X_shawarma), np.ones(N_shawarma)) + a - 1) / (N_shawarma + a + b - 2)

        pi = [N_pizza/N, N_sushi/N, N_shawarma/N]

        return pi, theta

    def _training_subset(self, X, y):
        percent_split = self.split
        X_random = np.array(X.copy())
        y_random = np.array(y.copy())

        p = np.random.permutation(len(y_random))
        X_random, y_random = X_random[p], y_random[p]

        slice1 = int(np.floor(percent_split * len(y_random) / 100))
        return X_random[:slice1], y_random[:slice1]

    def _single_prediction(self, X, pi, theta):
      results = []

      # Use log-probabilities instead of exponentiating
      log_pi = np.log(pi)

      log_pizza = np.matmul(X, np.log(theta[:, 0])) + np.matmul(1 - X, np.log(1 - theta[:, 0])) + log_pi[0]
      results.append(log_pizza)

      log_sushi = np.matmul(X, np.log(theta[:, 1])) + np.matmul(1 - X, np.log(1 - theta[:, 1])) + log_pi[1]
      results.append(log_sushi)

      log_shawarma = np.matmul(X, np.log(theta[:, 2])) + np.matmul(1 - X, np.log(1 - theta[:, 2])) + log_pi[2]
      results.append(log_shawarma)

      # Compare log scores directly
      y = np.argmax(results, axis=0)
      return y

    def fit(self, X, y, sample_weight=None):
        pi_map = []
        theta_map = []
        N = self.N
        for i in range(N):
            X_batch, y_batch = self._training_subset(X, y)
            pi_map_temp, theta_map_temp = self._map_pi_theta(X_batch, y_batch)
            pi_map.append(pi_map_temp)
            theta_map.append(theta_map_temp)

        self.pi_map = np.mean(pi_map, axis=0)
        self.theta_map = np.mean(theta_map, axis=0)
        self.theta_map = np.clip(self.theta_map, 1e-9, 1 - 1e-9)
        return self

    def predict(self, X):
        N = self.N
        pi = self.pi_map
        theta = self.theta_map
        y_temp = self._single_prediction(X, pi, theta)

        y_map = ["Pizza" if x==0 else "Sushi" if x==1 else "Shawarma" for x in y_temp]
        return np.array(y_map)

    def get_params(self, deep=False):
        if deep:
            params = {}
            for parameter, value in self:
                params[parameter] = value
            return params
        else:
            return {"a": self.a, "b": self.b, "N": self.N, "split": self.split}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def save(self, filename: str):
        try:
            with open(filename, "wb") as f:
                params = {"pi": self.pi_map, "theta": self.theta_map}
                pickle.dump(params, f)
            print(f"Success! Naive Bayes exported to {filename}.")
        except pickle.PicklingError:
            print("Error: Naive Bayes could not be pickled. Double-check the types of data")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def load_pretrained(self, filename: str):
        try:
            with open(filename, "rb") as f:
                params = pickle.load(f)
                self.pi_map = params["pi"]
                self.theta_map = params["theta"]
            print(f"Success! Pre-trained Naive Bayes loaded from {filename}.")
        except pickle.UnpicklingError:
            print(f"Error: either {filename} is not a valid pickle file or is corrupted.")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def predict_proba(self, X):
        """Return probability estimates for each class (needed for soft voting)."""
        if not hasattr(self, 'pi_map') or not hasattr(self, 'theta_map'):
            raise ValueError("Model must be trained before calling predict_proba.")

        log_pi = np.log(self.pi_map)
        log_theta = np.log(self.theta_map)
        log_1_minus_theta = np.log(1 - self.theta_map)

        log_probs = np.zeros((X.shape[0], 3))  # 3 classes

        for i in range(3):
            log_prob_class = np.matmul(X, log_theta[:, i]) + np.matmul(1 - X, log_1_minus_theta[:, i]) + log_pi[i]
            log_probs[:, i] = log_prob_class

        # Convert log-probs to actual probabilities using softmax for stability
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        exp_log_probs = np.exp(log_probs - max_log_probs)
        probs = exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)

        return {
            "Pizza": probs[:, 0],
            "Sushi": probs[:, 1],
            "Shawarma": probs[:, 2]
        }

