import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import joblib
import os

# ---------------------------
# Load Dataset
# ---------------------------
data = pd.read_csv("data/heart.csv")

X = data.drop("HeartDisease", axis=1)
y = data["HeartDisease"]

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# ---------------------------
# Train Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# Feature Scaling (needed for LR & KNN)
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("model/saved", exist_ok=True)

joblib.dump(scaler, "model/saved/scaler.pkl")

# ---------------------------
# Metric Function
# ---------------------------
def evaluate_model(name, model, X_tr, X_te):
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    else:
        y_prob = None

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    joblib.dump(model, f"model/saved/{name.replace(' ', '_').lower()}.pkl")
    return metrics

results = []

# ---------------------------
# 1. Logistic Regression
# ---------------------------
lr = LogisticRegression(max_iter=1000)
results.append(evaluate_model(
    "Logistic Regression", lr, X_train_scaled, X_test_scaled
))

# ---------------------------
# 2. Decision Tree
# ---------------------------
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
results.append(evaluate_model(
    "Decision Tree", dt, X_train, X_test
))

# ---------------------------
# 3. KNN
# ---------------------------
knn = KNeighborsClassifier(n_neighbors=7)
results.append(evaluate_model(
    "KNN", knn, X_train_scaled, X_test_scaled
))

# ---------------------------
# 4. Naive Bayes
# ---------------------------
nb = GaussianNB()
results.append(evaluate_model(
    "Naive Bayes", nb, X_train, X_test
))

# ---------------------------
# 5. Random Forest
# ---------------------------
rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42
)
results.append(evaluate_model(
    "Random Forest", rf, X_train, X_test
))

# ---------------------------
# 6. XGBoost
# ---------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
results.append(evaluate_model(
    "XGBoost", xgb_model, X_train, X_test
))

# ---------------------------
# Save Metrics
# ---------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("model/saved/model_metrics.csv", index=False)

print("\nModel Training Completed Successfully!\n")
print(results_df)
