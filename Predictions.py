import pandas as pd
import numpy as np
import pickle
import re
import json

# Preprocessing Parameters
numerical_columns = [
    "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
    "Q2: How many ingredients would you expect this food item to contain?",
    "Q4: How much would you expect to pay for one serving of this food item?"
]

text_cols = [
    "Q3: In what setting would you expect this food to be served? Please check all that apply",
    "Q5: What movie do you think of when thinking of this food item?",
    "Q6: What drink would you pair with this food item?",
    "Q7: When you think about this food item, who does it remind you of?"
]

q8_col = "Q8: How much hot sauce would you add to this food item?"
hot_sauce_map = {
    "A little (mild)": "Mild",
    "A moderate amount (medium)": "Medium",
    "A lot (hot)": "Hot",
    "I will have some of this food item with my hot sauce": "Medium"
}

# Preprocessing 
def extract_numeric(value):
    if pd.isnull(value):
        return None
    numbers = re.findall(r"\d+\.?\d*", str(value))
    return np.mean([float(n) for n in numbers]) if numbers else None

def simple_bow(df, column_name, prefix):
    vocab = set()
    tokenized = []
    for text in df[column_name]:
        tokens = re.findall(r'\b\w+\b', text.lower())
        tokenized.append(tokens)
        vocab.update(tokens)
    vocab = sorted(vocab)
    vocab_index = {word: i for i, word in enumerate(vocab)}
    bow_matrix = np.zeros((len(df), len(vocab)), dtype=int)
    for i, tokens in enumerate(tokenized):
        for token in tokens:
            if token in vocab_index:
                bow_matrix[i, vocab_index[token]] = 1
    return pd.DataFrame(bow_matrix, columns=[f"{prefix}_{word}" for word in vocab])

def preprocess(df):
    df = df.copy()
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    for col in numerical_columns:
        df[col] = df[col].apply(extract_numeric)
        df[col] = df[col].fillna(df[col].mean())

    for col in text_cols:
        df[col] = df[col].fillna("none").str.lower().str.strip()

    bow_frames = []
    for i, col in enumerate(text_cols):
        bow_df = simple_bow(df, col, f"Q{i+3}")
        bow_frames.append(bow_df)

    df = pd.concat([df] + bow_frames, axis=1)
    df.drop(columns=text_cols, inplace=True)

    df["Q8_cleaned"] = df[q8_col].map(hot_sauce_map).fillna("None")
    for category in df["Q8_cleaned"].unique():
        df[f"Q8_cleaned_{category}"] = (df["Q8_cleaned"] == category).astype(int)
    df.drop(columns=[q8_col, "Q8_cleaned"], inplace=True)

    df.fillna(0, inplace=True)
    for col in df.columns:
        if df[col].dtype in ["float64", "bool"]:
            df[col] = df[col].astype(int)

    with open("final_feature_names.json", "r") as f:
        expected_columns = json.load(f)

    missing_cols = [col for col in expected_columns if col not in df.columns]
    for col in missing_cols:
        df[col] = 0
    df = df[expected_columns]
    return df.astype(np.float64).values

# Prediction 
def predict_all(csv_path):
    df = pd.read_csv(csv_path)
    X = preprocess(df)

    with open("final_nb_model.pkl", "rb") as f:
        nb_params = pickle.load(f)

    with open("final_lr_model.pkl", "rb") as f:
        lr_params = pickle.load(f)

    with open("final_rf_model.pkl", "rb") as f:
        rf_trees = pickle.load(f)

    def nb_predict_proba(X, pi, theta):
        log_pi = np.log(pi)
        log_theta = np.log(theta)
        log_1_theta = np.log(1 - theta)
        log_probs = np.zeros((X.shape[0], 3))
        for i in range(3):
            log_probs[:, i] = X @ log_theta[:, i] + (1 - X) @ log_1_theta[:, i] + log_pi[i]
        exp = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return {
            "Pizza": probs[:, 0],
            "Sushi": probs[:, 1],
            "Shawarma": probs[:, 2]
        }

    nb_probs = nb_predict_proba(X, np.array(nb_params["pi"]), np.array(nb_params["theta"]))

    def lr_predict_proba(X, classifiers):
        probs = {}
        class_order = list(classifiers.keys())
        prob_matrix = np.zeros((X.shape[0], len(class_order)))
        for i, cls in enumerate(class_order):
            model = classifiers[cls]
            X_scaled = (X - np.array(model["X_min"])) / (np.array(model["X_max"]) - np.array(model["X_min"]) + 1e-8)
            z = np.clip(X_scaled @ np.array(model["weights"]) + model["bias"], -500, 500)
            prob_matrix[:, i] = 1 / (1 + np.exp(-z))
            probs[cls] = prob_matrix[:, i]
        return probs

    lr_probs = lr_predict_proba(X, lr_params)

    def predict_tree(tree, x):
        while isinstance(tree, dict) and 'label' not in tree:
            feature = tree["feature"]
            value = tree["value"]
            if x[feature] == value:
                tree = tree["left"]
            else:
                tree = tree["right"]
        return tree["label"]

    def rf_predict_proba(X, trees):
        class_labels = ["Pizza", "Sushi", "Shawarma"]
        label_to_idx = {l: i for i, l in enumerate(class_labels)}
        votes = np.zeros((X.shape[0], 3))
        for tree in trees:
            preds = [predict_tree(tree, x) for x in X]
            for i, p in enumerate(preds):
                votes[i][label_to_idx[p]] += 1
        probs = votes / len(trees)
        return {
            "Pizza": probs[:, 0],
            "Sushi": probs[:, 1],
            "Shawarma": probs[:, 2]
        }

    rf_probs = rf_predict_proba(X, rf_trees)

    # Soft Voting 
    labels = ["Pizza", "Sushi", "Shawarma"]
    weights = {"nb": 0.1, "lr": 0.8, "rf": 0.1}
    final_probs = {}
    for label in labels:
        final_probs[label] = (
            weights["nb"] * nb_probs[label] +
            weights["lr"] * lr_probs[label] +
            weights["rf"] * rf_probs[label]
        )

    predictions = []
    for i in range(X.shape[0]):
        label_scores = {label: final_probs[label][i] for label in labels}
        best = max(label_scores, key=label_scores.get)
        predictions.append(best)

    return predictions
