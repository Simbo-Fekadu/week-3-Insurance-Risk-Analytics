import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

try:
    # Load data (adjust sep based on inspection)
    df = pd.read_csv('../data/MachineLearningRating_v3.txt', sep='|')
except FileNotFoundError:
    print("Error: MachineLearningRating_v3.txt not found in data/.")
    exit()

# Preprocessing
df['HasClaim'] = df['TotalClaims'] > 0
df['ClaimSeverity'] = df['TotalClaims'].where(df['HasClaim'], np.nan)
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

# Select features and handle missing values
features = ['TotalPremium', 'CustomValueEstimate', 'Province', 'PostalCode', 'Gender', 'VehicleType']
numeric_features = ['TotalPremium', 'CustomValueEstimate']
categorical_features = ['Province', 'PostalCode', 'Gender', 'VehicleType']

# Encode categorical variables
df_encoded = pd.get_dummies(df[features], columns=categorical_features, drop_first=True)

# Handle missing values
df_encoded[numeric_features] = df_encoded[numeric_features].fillna(df_encoded[numeric_features].mean())

# 1. Logistic Regression: Predict Claim Frequency (HasClaim)
X = df_encoded
y_freq = df['HasClaim'].fillna(False)
X_train, X_test, y_train, y_test = train_test_split(X, y_freq, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_freq = log_reg.predict(X_test)

print("Logistic Regression (Claim Frequency):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_freq):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_freq):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_freq):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_freq))

# 2. Random Forest: Predict Claim Frequency
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf_freq = rf_clf.predict(X_test)

print("\nRandom Forest (Claim Frequency):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_freq):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf_freq):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf_freq):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_rf_freq))

# Feature Importance (Random Forest)
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_clf.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Top 10 Feature Importance (Claim Frequency)')
plt.savefig('../docs/feature_importance_frequency.png')
plt.show()

# 3. Linear Regression: Predict Claim Severity
df_severity = df[df['HasClaim']].copy()
X_severity = pd.get_dummies(df_severity[features], columns=categorical_features, drop_first=True)
X_severity[numeric_features] = X_severity[numeric_features].fillna(X_severity[numeric_features].mean())
y_severity = df_severity['ClaimSeverity'].fillna(df_severity['ClaimSeverity'].mean())

X_train_sev, X_test_sev, y_train_sev, y_test_sev = train_test_split(X_severity, y_severity, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train_sev, y_train_sev)
y_pred_sev = lin_reg.predict(X_test_sev)

print("\nLinear Regression (Claim Severity):")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_sev, y_pred_sev)):.2f}")
print(f"R²: {r2_score(y_test_sev, y_pred_sev):.4f}")

# 4. Random Forest: Predict Claim Severity
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train_sev, y_train_sev)
y_pred_rf_sev = rf_reg.predict(X_test_sev)

print("\nRandom Forest (Claim Severity):")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_sev, y_pred_rf_sev)):.2f}")
print(f"R²: {r2_score(y_test_sev, y_pred_rf_sev):.4f}")

# Plot Predictions vs Actuals (Claim Severity)
plt.figure(figsize=(8, 6))
plt.scatter(y_test_sev, y_pred_rf_sev, alpha=0.5)
plt.plot([y_test_sev.min(), y_test_sev.max()], [y_test_sev.min(), y_test_sev.max()], 'r--')
plt.xlabel('Actual Claim Severity')
plt.ylabel('Predicted Claim Severity')
plt.title('Random Forest: Claim Severity Predictions')
plt.savefig('../docs/pred_vs_actual_severity.png')
plt.show()

# 5. XGBoost: Predict Margin
X_margin = df_encoded
y_margin = df['Margin'].fillna(df['Margin'].mean())
X_train_margin, X_test_margin, y_train_margin, y_test_margin = train_test_split(X_margin, y_margin, test_size=0.2, random_state=42)

xgb_reg = XGBRegressor(random_state=42)
xgb_reg.fit(X_train_margin, y_train_margin)
y_pred_margin = xgb_reg.fit(X_test_margin)

print("\nXGBoost (Margin):")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_margin, y_pred_margin)):.2f}")
print(f"R²: {r2_score(y_test_margin, y_pred_margin):.4f}")