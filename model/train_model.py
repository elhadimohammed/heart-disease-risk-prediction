import json
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# -----------------------------
# 1) File paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "logistic_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

# -----------------------------
# 2) Load dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)

# Normalize column names
df.columns = df.columns.str.strip().str.lower()

print("Columns found in dataset:")
print(df.columns.tolist())

# -----------------------------
# 3) Fix known column name differences
# -----------------------------
# Your dataset uses 'thalch' instead of standard UCI 'thalach'
rename_map = {
    "thalch": "thalach"
}
df = df.rename(columns=rename_map)

# -----------------------------
# 4) Define expected columns
# -----------------------------
expected_columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "num"
]

missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    raise ValueError(
        f"Missing required columns in dataset: {missing_columns}\n"
        f"Found columns: {df.columns.tolist()}"
    )

# Keep only required columns and ignore extras like id, dataset
df = df[expected_columns].copy()

print("\nColumns used for training:")
print(df.columns.tolist())

# -----------------------------
# 5) Handle missing values
# -----------------------------
df = df.replace("?", pd.NA)

# Convert all columns to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nMissing values per column:")
print(df.isnull().sum())

# -----------------------------
# 6) Create binary target
# -----------------------------
# num = 0  -> no disease
# num > 0  -> disease present
df["target"] = df["num"].apply(lambda x: 0 if x == 0 else 1)

# Drop original target column
df.drop(columns=["num"], inplace=True)

# -----------------------------
# 7) Features and target
# -----------------------------
X = df.drop(columns=["target"])
y = df["target"]

feature_columns = X.columns.tolist()

# -----------------------------
# 8) Feature types
# -----------------------------
numeric_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

# -----------------------------
# 9) Preprocessing
# -----------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# 10) Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 11) Model pipeline
# -----------------------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# -----------------------------
# 12) Train model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 13) Evaluate model
# -----------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    "precision": round(float(precision_score(y_test, y_pred)), 4),
    "recall": round(float(recall_score(y_test, y_pred)), 4),
    "f1_score": round(float(f1_score(y_test, y_pred)), 4),
    "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
}

# -----------------------------
# 14) Save outputs
# -----------------------------
joblib.dump(model, MODEL_PATH)
joblib.dump(feature_columns, FEATURES_PATH)

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

# -----------------------------
# 15) Print results
# -----------------------------
print("\nTraining completed successfully.")
print(f"Model saved to: {MODEL_PATH}")
print(f"Feature columns saved to: {FEATURES_PATH}")
print(f"Metrics saved to: {METRICS_PATH}")

print("\nEvaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")