import pyodbc

# -----------------------------
# SQL Server connection settings
# -----------------------------
SERVER = "localhost\SQLEXPRESS"
DATABASE = "HeartDiseaseDB"

# If you use Windows Authentication:
CONNECTION_STRING = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"Trusted_Connection=yes;"
)

# If you use SQL Server username/password instead, use this instead:
# USERNAME = "sa"
# PASSWORD = "your_password"
# CONNECTION_STRING = (
#     f"DRIVER={{ODBC Driver 17 for SQL Server}};"
#     f"SERVER={SERVER};"
#     f"DATABASE={DATABASE};"
#     f"UID={USERNAME};"
#     f"PWD={PASSWORD};"
# )

def get_connection():
    return pyodbc.connect(CONNECTION_STRING)

def save_prediction(data):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        INSERT INTO predictions (
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal,
            predicted_probability, prediction_result
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    values = (
        data["age"],
        data["sex"],
        data["cp"],
        data["trestbps"],
        data["chol"],
        data["fbs"],
        data["restecg"],
        data["thalach"],
        data["exang"],
        data["oldpeak"],
        data["slope"],
        data["ca"],
        data["thal"],
        data["predicted_probability"],
        data["prediction_result"]
    )

    cursor.execute(query, values)
    conn.commit()
    conn.close()

def get_all_predictions():
    conn = get_connection()
    cursor = conn.cursor()

    query = """
        SELECT
            prediction_id, age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal,
            predicted_probability, prediction_result, created_at
        FROM predictions
        ORDER BY created_at DESC
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    return rows