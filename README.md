# ML Classifier 

## Overview
This project implements a **custom machine learning classification system built entirely from scratch**, without using scikit-learn or deep learning libraries. The objective is to classify survey responses into one of three categories: **Pizza, Sushi, or Shawarma**.

The final system combines **Logistic Regression, Naive Bayes, and Random Forest** models using a **weighted soft-voting ensemble**, achieving strong and consistent performance across all classes.

> This project was developed under strict implementation constraints for my upper-year university machine learning course and later refined.

---

## Key Highlights
- Fully custom implementations of:
  - Multiclass Logistic Regression (One-vs-Rest, softmax)
  - Naive Bayes with MAP estimation and Beta priors
  - Random Forest with custom decision trees and bagging
- Weighted soft-voting ensemble for final predictions
- Manual feature engineering for mixed data types
- Stratified train/validation splitting to preserve class balance
- Final ensemble accuracy: **~89%**

---

## Data Processing & Feature Engineering
The dataset contains a mixture of **free-form text responses, multi-select answers, and numeric inputs**. To handle this effectively:

- Text-response questions were encoded using a **binary bag-of-words** approach
- One categorical question was **one-hot encoded**
- Numeric responses were cleaned and converted to numerical values
- All final feature matrices contained only **integers and floats**
- Multiple preprocessing strategies were explored and evaluated

---

## Models

### Logistic Regression
- Custom One-vs-Rest multiclass implementation
- Feature scaling for numerical stability
- L2 regularization to reduce overfitting
- Tuned learning rate, regularization strength, and number of epochs
- Produced stable, interpretable, and high-performing results
- Served as the **primary contributor** in the ensemble

---

### Random Forest
- Custom implementation using multiple decision trees
- Trained using bootstrap sampling (bagging)
- Random feature selection at each split to reduce correlation
- Gini impurity used for computational efficiency
- No maximum depth, minimum samples to split = 2

---

### Naive Bayes
- MAP estimation with Beta priors
- Designed to handle free-form text effectively
- Bag-of-words feature extraction
- Bagging used to improve consistency across trials
- Tuned prior parameters and sampling percentages

---

## Ensemble Model
The final classifier uses **soft voting**, combining probability outputs from each model:

| Model | Weight |
|------|--------|
| Logistic Regression | 0.8 |
| Naive Bayes | 0.1 |
| Random Forest | 0.1 |

This ensemble outperformed each individual model.

---

## Evaluation
- Stratified 70/30 train-validation split
- Accuracy used as the primary evaluation metric
- Precision and recall examined per class to detect bias
- Final test accuracy on held-out data: **~0.895**

The model performs consistently across all three classes, with balanced precision and recall.

