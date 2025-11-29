# Predictive-Maintenance-
Predictive Maintenance using Machine Learning
# 🧠 Predictive Maintenance using Machine Learning

## 📋 Project Overview
This project develops a **predictive maintenance system** capable of forecasting machine failures based on industrial sensor readings. Using supervised machine learning models, it predicts potential equipment breakdowns before they occur, enabling proactive maintenance and minimizing downtime.

The workflow covers the entire pipeline — from **data loading and visualization** to **model training, tuning, and ensemble deployment**.

---

## 🚀 Key Features
- **Raw Data Visualization:** Bar charts showing raw sensor values and failure trends before preprocessing.
- **Data Cleaning & Preprocessing:** Automatic handling of duplicates, missing values, and outliers.
- **Exploratory Data Analysis (EDA):**
  - Feature-wise correlation with failure occurrence.
  - Combined ascending bar chart to identify top failure-causing features.
- **Model Training & Tuning:**
  - Algorithms used: Logistic Regression, Decision Tree, SVM, Random Forest, XGBoost, Gradient Boosting, Neural Network (Keras)
  - Hyperparameter tuning via `GridSearchCV`.
- **Model Evaluation:**
  - Metrics: Accuracy, F1-Score, ROC-AUC, Confusion Matrix.
  - ROC curve comparison for tuned models.
- **Stacking Ensemble:**
  - Combines multiple base learners for best performance.
  - Logistic Regression meta-learner for final decision.
- **Visualization:**
  - Failure counts, correlation heatmaps, and feature importance charts.
- **Deployment Ready:**
  - Includes model saving using `joblib` and Flask/FastAPI integration for real-time predictions.

---

## 🧩 Dataset Information
The dataset used is available on **Kaggle**:

**Name:** Predictive Maintenance Dataset  
**Source:** [https://www.kaggle.com/code/balamuruganmaharajan/predictive-maintenance)

**Main Features:**
| Feature | Description |
|----------|--------------|
| Air temperature [K] | Air temperature in Kelvin |
| Process temperature [K] | Process temperature in Kelvin |
| Rotational speed [rpm] | Rotational speed of the tool |
| Torque [Nm] | Torque applied on the tool |
| Tool wear [min] | Time in minutes the tool has been used |
| Target / Machine failure | Binary target variable (1 = Failure, 0 = Normal) |

---

## 🧹 Data Preprocessing Steps
1. Load raw CSV data from Kaggle.  
2. Visualize **raw data bar chart** for all five main features.  
3. Remove unnecessary columns (`UDI`, `Product ID`, etc.).  
4. Handle missing values and duplicates.  
5. Cap extreme outliers using the IQR method.  
6. One-hot encode categorical columns.  
7. Normalize features using `StandardScaler`.

---

## 📈 Exploratory Data Analysis (EDA)
- **Raw feature plots:** Bar graphs of raw sensor data vs machine failure (before cleaning).  
- **Failure frequency chart:** Combined ascending bar chart showing which feature contributes most to failures.  
- **Correlation plots:** Feature correlation with machine failure and overall heatmap visualization.

---

## ⚙️ Model Training
Models implemented:
- Logistic Regression  
- Decision Tree Classifier  
- Support Vector Machine (SVM)  
- Random Forest Classifier  
- Gradient Boosting Classifier  
- XGBoost Classifier  
- Neural Network (MLPClassifier, TensorFlow Sequential Model)

Each model was trained using `train_test_split()` with a test size of 20%.

---

## 🔍 Hyperparameter Tuning
- Performed for **Decision Tree**, **SVM**, and **Gradient Boosting** using `GridSearchCV`.  
- Selected based on **F1-score** performance.  
- Evaluated using cross-validation for robustness.

---

## 🏆 Ensemble Stacking
Stacking combines multiple tuned models into a final **meta-learner**:
```python
StackingClassifier(
    estimators=[('SVM', best_svm), ('GB', best_gb), ('RF', best_rf)],
    final_estimator=LogisticRegression()
)
