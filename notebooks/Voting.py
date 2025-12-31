import numpy as np

class VotingClassifier():
    def __init__(self, estimators, voting) -> None:
        self.models = {}
        self.num_models = 0
        for item in estimators:
            self.models[item[0]] = item[1]
            self.num_models += 1

    def fit(self, X, y):
        for item in self.models:
            self.models[item].fit(X, y)
        return self

    def argmax_with_tie_breaking(self, array):
        result = []
        for item in np.array(array).T:
            max_value = np.max(item)
            max_indices = np.where(item == max_value)[0]
            result.append(np.random.choice(max_indices))
        return np.array(result)

    def predict(self, X, random=True):
        predictions = []
        y_map_count = np.zeros([3, X.shape[0]])
        for key in self.models:
            model = self.models[key]
            y_temp = model.predict(X)
            for i in range(y_temp.shape[0]):
                value = y_temp[i]
                if value == "Pizza":
                    y_map_count[0][i] += 1
                if value == "Sushi":
                    y_map_count[1][i] += 1
                if value == "Shawarma":
                    y_map_count[2][i] += 1
        if random:
            y_map = self.argmax_with_tie_breaking(y_map_count)
        else:
            y_map = np.argmax(y_map_count, axis = 0)
        y_map = ["Pizza" if x==0 else "Sushi" if x==1 else "Shawarma" for x in y_map]
        return np.array(y_map)
  
    
    def save(self, filename: str):
        for key in self.models:
            model = self.models[key]
            model.save(key + "_" + filename)

    def load_pretrained(self, filename: str):
        for key in self.models:
            model = self.models[key]
            model.load_pretrained(key + "_" + filename)