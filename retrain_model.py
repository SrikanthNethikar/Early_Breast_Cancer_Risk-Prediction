import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("breast_cancer_early_risk.csv")

# 🧠 Update if needed
target_column = "diagnosis_label"

# Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# 🔁 Convert categorical columns to numeric using one-hot encoding
X_encoded = pd.get_dummies(X)

feature_columns = X_encoded.columns.tolist()
joblib.dump(feature_columns, "feature_columns.pkl")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "breast_cancer_risk_model.pkl")

# Save test sets (for SHAP if needed)
X_test.to_csv("x_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("✅ Model retrained and saved successfully with encoded features!")