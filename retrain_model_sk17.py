import os  # ✅ Add this
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv("breast_cancer_early_risk.csv")

# Separate features and target
X = df.drop("diagnosis_label", axis=1)
y = df["diagnosis_label"]

# Encode categorical features
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('category').cat.codes

if y.dtype == 'object':
    y = y.astype('category').cat.codes

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "breast_cancer_risk_model.pkl")

# ✅ Show current folder and save test sets
print("Saving to:", os.getcwd())
X_test.to_csv("x_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("✅ Model retrained and saved successfully using scikit-learn 1.7.0")