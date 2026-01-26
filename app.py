import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# =====================
# App UI
# =====================
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("❤️ Heart Disease Prediction – ML Model Comparison")

# =====================
# Load models
# =====================
@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("model/saved/logistic.pkl"),
        "Decision Tree": joblib.load("model/saved/decision_tree.pkl"),
        "KNN": joblib.load("model/saved/knn.pkl"),
        "Naive Bayes": joblib.load("model/saved/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/saved/random_forest.pkl"),
        "XGBoost": joblib.load("model/saved/xgboost.pkl"),
    }

models = load_models()

# =====================
# Sidebar
# =====================
st.sidebar.header("Controls")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# =====================
# Prediction
# =====================
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Drop target if user uploads full dataset
    if "HeartDisease" in df.columns:
        y_true = df["HeartDisease"]
        X_test = df.drop(columns=["HeartDisease"])
    else:
        st.error("❌ Uploaded file must contain HeartDisease column for evaluation")
        st.stop()

    model = models[selected_model]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # =====================
    # Metrics
    # =====================
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # =====================
    # Display Metrics
    # =====================
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(acc, 3))
    col2.metric("AUC", round(auc, 3))
    col3.metric("MCC", round(mcc, 3))

    col4, col5, col6 = st.columns(3)
    col4.metric("Precision", round(prec, 3))
    col5.metric("Recall", round(rec, 3))
    col6.metric("F1 Score", round(f1, 3))

    # =====================
    # Confusion Matrix
    # =====================
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    st.write(cm)
