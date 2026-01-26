import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction – ML Comparison",
    layout="wide"
)

st.title("💓 Heart Disease Prediction – ML Model Comparison")
st.write(
    """
    This Streamlit application allows users to upload test data,
    select a trained machine learning model, and view prediction
    performance using multiple evaluation metrics.
    """
)

# --------------------------------------------------
# Load Models and Scaler
# --------------------------------------------------
@st.cache_resource
def load_resources():
    models = {
        "Logistic Regression": joblib.load("model/saved/logistic_regression.pkl"),
        "Decision Tree": joblib.load("model/saved/decision_tree.pkl"),
        "KNN": joblib.load("model/saved/knn.pkl"),
        "Naive Bayes": joblib.load("model/saved/naive_bayes.pkl"),
        "Random Forest": joblib.load("model/saved/random_forest.pkl"),
        "XGBoost": joblib.load("model/saved/xgboost.pkl")
    }
    scaler = joblib.load("model/saved/scaler.pkl")
    metrics_df = pd.read_csv("model/saved/model_metrics.csv")
    return models, scaler, metrics_df

models, scaler, metrics_df = load_resources()

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("🔧 Controls")

model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

# --------------------------------------------------
# Display Stored Metrics
# --------------------------------------------------
st.subheader("📊 Model Performance Summary (Test Set)")

selected_metrics = metrics_df[metrics_df["Model"] == model_name]

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", round(selected_metrics["Accuracy"].values[0], 3))
col1.metric("AUC", round(selected_metrics["AUC"].values[0], 3))

col2.metric("Precision", round(selected_metrics["Precision"].values[0], 3))
col2.metric("Recall", round(selected_metrics["Recall"].values[0], 3))

col3.metric("F1 Score", round(selected_metrics["F1"].values[0], 3))
col3.metric("MCC", round(selected_metrics["MCC"].values[0], 3))

# --------------------------------------------------
# Prediction on Uploaded Data
# --------------------------------------------------
if uploaded_file:
    st.subheader("🧪 Model Evaluation on Uploaded Test Data")

    test_data = pd.read_csv(uploaded_file)

    if "HeartDisease" not in test_data.columns:
        st.error("Uploaded CSV must contain 'HeartDisease' target column.")
    else:
        X_test = test_data.drop("HeartDisease", axis=1)
        y_test = test_data["HeartDisease"]

        model = models[model_name]

        # Scale when required
        if model_name in ["Logistic Regression", "KNN"]:
            X_test_processed = scaler.transform(X_test)
        else:
            X_test_processed = X_test

        y_pred = model.predict(X_test_processed)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_processed)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.markdown("### 📈 Evaluation Metrics")

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", round(acc, 3))
        c1.metric("AUC", round(auc, 3) if auc else "N/A")

        c2.metric("Precision", round(prec, 3))
        c2.metric("Recall", round(rec, 3))

        c3.metric("F1 Score", round(f1, 3))
        c3.metric("MCC", round(mcc, 3))

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------
        st.markdown("### 🔍 Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        st.pyplot(fig)

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------
        st.markdown("### 📄 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3))

else:
    st.info("👈 Upload a CSV file from the sidebar to evaluate a model.")
