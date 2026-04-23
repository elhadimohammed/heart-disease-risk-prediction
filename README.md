# heart-disease-risk-prediction

A web-based **Heart Disease Risk Prediction System** built using **Python (Flask)** for the backend, **HTML/CSS/JavaScript** for the frontend, **SQL Server** for database storage, and **Logistic Regression** from **scikit-learn** for machine learning.

This project predicts the probability of heart disease based on clinical patient data and classifies the result into **Low Risk**, **Medium Risk**, or **High Risk**.

> **Note:** This project is developed for **educational purposes only** and is **not** intended to be used as a real medical diagnosis tool.

---

## 📌 Project Overview

The goal of this project is to create a complete machine learning-powered web application that:

- accepts patient clinical input through a web form
- uses a trained Logistic Regression model to predict heart disease risk
- displays the prediction result in a user-friendly format
- stores prediction history in a SQL Server database
- allows users to review previously saved predictions

This system uses the **UCI Heart Disease dataset** as the training dataset.

---

## 🚀 Features

- Web-based heart disease risk prediction
- User-friendly input form
- Logistic Regression model using scikit-learn
- Probability-based risk classification:
  - **Low Risk**
  - **Medium Risk**
  - **High Risk**
- SQL Server integration for storing predictions
- Prediction history page
- Human-readable display of coded medical values
- Clean and responsive UI

---

## 🛠️ Technologies Used

### Backend
- Python
- Flask

### Frontend
- HTML
- CSS
- JavaScript

### Database
- SQL Server
- pyodbc

### Machine Learning
- scikit-learn
- pandas
- numpy
- joblib

---

## 📂 Project Structure

```text
heart-disease-risk-predictor/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── heart.csv
│
├── database/
│   ├── db.py
│   └── schema.sql
│
├── model/
│   ├── train_model.py
│   ├── logistic_model.pkl
│   ├── feature_columns.pkl
│   └── metrics.json
│
├── templates/
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   └── history.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
└── utils/
``
