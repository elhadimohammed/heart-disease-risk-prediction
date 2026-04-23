IF OBJECT_ID('predictions', 'U') IS NOT NULL
    DROP TABLE predictions;
GO

CREATE TABLE predictions (
    prediction_id INT PRIMARY KEY IDENTITY(1,1),

    age INT NOT NULL,
    sex INT NOT NULL,
    cp INT NOT NULL,
    trestbps FLOAT NOT NULL,
    chol FLOAT NOT NULL,
    fbs INT NOT NULL,
    restecg INT NOT NULL,
    thalach FLOAT NOT NULL,
    exang INT NOT NULL,
    oldpeak FLOAT NOT NULL,
    slope INT NOT NULL,
    ca INT NOT NULL,
    thal INT NOT NULL,

    predicted_probability FLOAT NOT NULL,
    prediction_result VARCHAR(20) NOT NULL,

    created_at DATETIME DEFAULT GETDATE()
);
GO