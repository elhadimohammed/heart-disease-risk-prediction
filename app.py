import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from database.db import save_prediction, get_all_predictions

app = Flask(__name__)

# -----------------------------
# Load model and feature columns
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "logistic_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_columns.pkl")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

# -----------------------------
# Human-readable mappings
# -----------------------------
SEX_MAP = {
    0: "Female",
    1: "Male"
}

CP_MAP = {
    1: "Typical Angina",
    2: "Atypical Angina",
    3: "Non-anginal Pain",
    4: "Asymptomatic"
}

FBS_MAP = {
    0: "False",
    1: "True"
}

RESTECG_MAP = {
    0: "Normal",
    1: "ST-T Wave Abnormality",
    2: "Left Ventricular Hypertrophy"
}

EXANG_MAP = {
    0: "No",
    1: "Yes"
}

SLOPE_MAP = {
    1: "Upsloping",
    2: "Flat",
    3: "Downsloping"
}

THAL_MAP = {
    3: "Normal",
    6: "Fixed Defect",
    7: "Reversible Defect"
}

# -----------------------------
# Helper: risk label
# -----------------------------
def get_risk_label(probability):
    if probability < 0.34:
        return "Low Risk"
    elif probability < 0.67:
        return "Medium Risk"
    else:
        return "High Risk"

# -----------------------------
# Helper: decode single patient input
# -----------------------------
def decode_input_data(input_data):
    return {
        "age": input_data["age"],
        "sex": SEX_MAP.get(input_data["sex"], input_data["sex"]),
        "cp": CP_MAP.get(input_data["cp"], input_data["cp"]),
        "trestbps": input_data["trestbps"],
        "chol": input_data["chol"],
        "fbs": FBS_MAP.get(input_data["fbs"], input_data["fbs"]),
        "restecg": RESTECG_MAP.get(input_data["restecg"], input_data["restecg"]),
        "thalach": input_data["thalach"],
        "exang": EXANG_MAP.get(input_data["exang"], input_data["exang"]),
        "oldpeak": input_data["oldpeak"],
        "slope": SLOPE_MAP.get(input_data["slope"], input_data["slope"]),
        "ca": input_data["ca"],
        "thal": THAL_MAP.get(input_data["thal"], input_data["thal"])
    }

# -----------------------------
# Helper: format history rows
# -----------------------------
def format_history_rows(rows):
    formatted = []

    for row in rows:
        formatted.append({
            "prediction_id": row.prediction_id,
            "age": row.age,
            "sex": SEX_MAP.get(row.sex, row.sex),
            "cp": CP_MAP.get(row.cp, row.cp),
            "trestbps": row.trestbps,
            "chol": row.chol,
            "fbs": FBS_MAP.get(row.fbs, row.fbs),
            "restecg": RESTECG_MAP.get(row.restecg, row.restecg),
            "thalach": row.thalach,
            "exang": EXANG_MAP.get(row.exang, row.exang),
            "oldpeak": row.oldpeak,
            "slope": SLOPE_MAP.get(row.slope, row.slope),
            "ca": row.ca,
            "thal": THAL_MAP.get(row.thal, row.thal),
            "predicted_probability": round(float(row.predicted_probability) * 100, 2),
            "prediction_result": row.prediction_result,
            "created_at": row.created_at
        })

    return formatted

# -----------------------------
# Home page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Prediction form page
# -----------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            input_data = {
                "age": int(request.form["age"]),
                "sex": int(request.form["sex"]),
                "cp": int(request.form["cp"]),
                "trestbps": float(request.form["trestbps"]),
                "chol": float(request.form["chol"]),
                "fbs": int(request.form["fbs"]),
                "restecg": int(request.form["restecg"]),
                "thalach": float(request.form["thalach"]),
                "exang": int(request.form["exang"]),
                "oldpeak": float(request.form["oldpeak"]),
                "slope": int(request.form["slope"]),
                "ca": int(request.form["ca"]),
                "thal": int(request.form["thal"])
            }

            # Create DataFrame in the same feature order used in training
            input_df = pd.DataFrame([input_data])
            input_df = input_df[feature_columns]

            # Predict
            probability = float(model.predict_proba(input_df)[0][1])
            risk_label = get_risk_label(probability)

            # Save raw numeric data to DB
            db_data = input_data.copy()
            db_data["predicted_probability"] = round(probability, 4)
            db_data["prediction_result"] = risk_label
            save_prediction(db_data)

            # Decode values for friendly display
            display_data = decode_input_data(input_data)

            return render_template(
                "result.html",
                probability=round(probability * 100, 2),
                risk_label=risk_label,
                display_data=display_data
            )

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    return render_template("predict.html")

# -----------------------------
# History page
# -----------------------------
@app.route("/history")
def history():
    rows = get_all_predictions()
    formatted_rows = format_history_rows(rows)
    return render_template("history.html", rows=formatted_rows)

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)