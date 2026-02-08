# ML_Ass2
Heart Disease Classification using Machine Learning
a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a patient has heart disease based on clinical attributes. The study evaluates different classification algorithms and compares their performance using multiple evaluation metrics.

b. Dataset Description

Dataset Source: UCI Machine Learning Repository / Kaggle

Total Instances: 918

Total Features: 12 clinical attributes

Target Variable: HeartDisease (0 = No disease, 1 = Disease)

The dataset contains demographic, clinical, and diagnostic features such as Age, Sex, Chest Pain Type, Resting Blood Pressure, Cholesterol level, ECG results, Exercise Angina, and ST slope.

| ML Model Name            | Accuracy | AUC   | Precision | Recall | F1 Score | MCC  |
| ------------------------ | -------- | ------| --------- | ------ | -------- | ---- |
| Logistic Regression      | 0.874    | 0.933  | 0.871    | 0.906  | 0.888    | 0.744|
| Decision Tree            | 1.0      | 1.0    | 1.0      | 1.0    | 1.0      | 1.0  |
| KNN                      | 0.894    | 0.965  | 0.888    | 0.925  | 0.906    | 0.786|
| Naive Bayes              | 0.862    | 0.921  | 0.873    | 0.878  | 0.875    | 0.72 |
| Random Forest (Ensemble) | 1.0      | 1.0    | 1.0      | 1.0    | 1.0      | 1.0  |
| XGBoost (Ensemble)       | 1.0      | 1.0    | 1.0      | 1.0    | 1.0      | 1.0  |

| ML Model Name            | Observation about Model Performance                                                         |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| Logistic Regression      | Performs well on linearly separable relationships and provides stable baseline performance. |
| Decision Tree            | Captures non-linear relationships but may overfit the training data.                        |
| KNN                      | Performance depends on the choice of K and scaling of features.                             |
| Naive Bayes              | Works efficiently but assumes feature independence which may reduce accuracy.               |
| Random Forest (Ensemble) | Provides better generalization by combining multiple trees and reducing overfitting.        |
| XGBoost (Ensemble)       | Achieves the best performance due to boosting and sequential error correction.              |

Repository Structure:
project-folder/
│-- app.py
│-- Models/train_model.py
│-- requirements.txt
│-- README.md
│-- model/saved
│     ├── logistic.pkl
│     ├── decision_tree.pkl
│     ├── knn.pkl
│     ├── naive_bayes.pkl
│     ├── random_forest.pkl
│     └── xgboost.pkl
Deployment

The application is deployed using Streamlit Community Cloud, providing an interactive web interface for heart disease prediction.
