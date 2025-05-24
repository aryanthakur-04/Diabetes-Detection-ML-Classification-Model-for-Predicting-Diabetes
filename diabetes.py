# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import shap
import warnings
warnings.filterwarnings('ignore')

# 2. Load the Dataset
data = pd.read_csv(r"C:\Users\thaku\Desktop\certification\diabetes.csv")
print("Shape of the dataset:", data.shape)
print(data.head())

# 3. Data Exploration
print("\nDataset Info:")
print(data.info())
print("\nSummary:")
print(data.describe())

# Check zero values in key features
columns_to_check = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print("\nZero values in important columns:")
print((data[columns_to_check] == 0).sum())

# Replace 0s with NaN and fill with median
data[columns_to_check] = data[columns_to_check].replace(0, np.nan)
data.fillna(data.median(), inplace=True)

# FEATURE ENGINEERING - BMI category
def bmi_cat(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'
data['BMI_cat'] = data['BMI'].apply(bmi_cat)

# FEATURE ENGINEERING - Age groups
def age_group(age):
    if age < 30:
        return 'Young'
    elif age < 50:
        return 'Middle-aged'
    else:
        return 'Senior'
data['Age_group'] = data['Age'].apply(age_group)

# One-hot encode the categorical features
data = pd.get_dummies(data, columns=['BMI_cat', 'Age_group'], drop_first=True)

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Features")
plt.show()

# Plot Outcome distribution
sns.countplot(x='Outcome', data=data)
plt.title("Count of Diabetes Cases (0 = No, 1 = Yes)")
plt.show()

# 4. Data Preprocessing
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train-Test Split (before scaling and SMOTE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE on training data only
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Feature Scaling
scaler = StandardScaler()
X_train_sm_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training with Hyperparameter Tuning for RF and XGBoost
lr = LogisticRegression(max_iter=1000, random_state=42)
svm = SVC(probability=True, random_state=42)
lr.fit(X_train_sm_scaled, y_train_sm)
svm.fit(X_train_sm_scaled, y_train_sm)

# Random Forest tuning
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='roc_auc', n_jobs=-1)
grid_rf.fit(X_train_sm_scaled, y_train_sm)
best_rf = grid_rf.best_estimator_
best_rf.fit(X_train_sm_scaled, y_train_sm)
print("Best RF params:", grid_rf.best_params_)

# XGBoost tuning
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
param_grid_xgb = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1]
}
grid_xgb = GridSearchCV(xgb, param_grid_xgb, cv=5, scoring='roc_auc', n_jobs=-1)
grid_xgb.fit(X_train_sm_scaled, y_train_sm)
best_xgb = grid_xgb.best_estimator_
best_xgb.fit(X_train_sm_scaled, y_train_sm)
print("Best XGB params:", grid_xgb.best_params_)

# 6. Evaluate Models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {model.__class__.__name__}')
    plt.show()

    # ROC Curve
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.show()

print("\nLogistic Regression Performance:")
evaluate_model(lr, X_test_scaled, y_test)
print("\nSVM Performance:")
evaluate_model(svm, X_test_scaled, y_test)
print("\nRandom Forest Performance:")
evaluate_model(best_rf, X_test_scaled, y_test)
print("\nXGBoost Performance:")
evaluate_model(best_xgb, X_test_scaled, y_test)

# 7. SHAP Explanation for best model (using XGBoost)
explainer = shap.Explainer(best_xgb)
shap_values = explainer(X_test_scaled)
X_test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
print("SHAP summary plot:")
shap.summary_plot(shap_values, X_test_df)
print("SHAP dependence plot for Glucose:")
shap.dependence_plot('Glucose', shap_values.values, X_test_df)

# 8. Risk Scorecard
original_data = pd.read_csv(r"C:\Users\thaku\Desktop\certification\diabetes.csv")  # Reload original data (raw, unscaled)
original_data[columns_to_check] = original_data[columns_to_check].replace(0, np.nan)
original_data.fillna(original_data.median(), inplace=True)
def calculate_risk_score(row):
    score = 0
    if row['Glucose'] > 150:
        score += 2
    elif row['Glucose'] > 120:
        score += 1
    if row['BMI'] > 30:
        score += 2
    elif row['BMI'] > 25:
        score += 1
    if row['Age'] > 50:
        score += 2
    elif row['Age'] > 35:
        score += 1
    if row['BloodPressure'] > 80:
        score += 1
    return score
original_data['RiskScore'] = original_data.apply(calculate_risk_score, axis=1)
def classify_risk(score):
    if score >= 5:
        return "High Risk"
    elif score >= 3:
        return "Moderate Risk"
    else:
        return "Low Risk"
original_data['RiskLevel'] = original_data['RiskScore'].apply(classify_risk)
print("\nSample Risk Scorecard Output:")
print(original_data[['Glucose', 'BMI', 'Age', 'BloodPressure', 'RiskScore', 'RiskLevel', 'Outcome']].head(10))

# Plot Risk Category Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='RiskLevel', data=original_data, palette='Set2')
plt.title("Patient Risk Categories")
plt.show()

# Compare Risk vs Actual Outcome
print("\nRisk Level vs Actual Diabetes Diagnosis:")
print(pd.crosstab(original_data['RiskLevel'], original_data['Outcome']))
