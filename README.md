# Car Price Prediction Web Application

This project is a machine learning–based web application that predicts the price of a car based on its specifications. It uses a trained Random Forest Regression model and a Flask web interface to provide real-time predictions.

# Project Overview

The system analyzes historical car data, performs preprocessing and feature encoding, trains multiple regression models, and deploys the best-performing model using Flask. Users can enter car details through a web form and instantly receive an estimated price.

# Technologies Used

Python

Flask

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn

HTML & CSS

Pickle (Model Serialization)

# Project Structure
carPricePrediction/
│
├── app.py
├── CarPricePrediction.ipynb
├── CarPrice_Assignment.csv
│
├── models/
│   ├── rf_model.pkl
│   ├── dt_model.pkl
│   ├── lr_model.pkl
│   └── feature_columns.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── README.md

# Machine Learning Pipeline

Load and explore dataset

Handle missing values and duplicates

Encode categorical features

Train-test split

Train models:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Evaluate using:

R² Score

MAE

RMSE

Save trained models and feature columns

Deploy best model using Flask

# Web Application Features

User-friendly input form

Accepts numerical & categorical car attributes

One-hot encoding aligned with training data

Real-time price prediction

Clean and simple UI

# How to Run the Project
1️⃣ Install required libraries
pip install flask pandas numpy scikit-learn matplotlib seaborn

2️⃣ Run the Flask app
python app.py

3️⃣ Open in browser
http://127.0.0.1:5000

# Input Parameters

The model uses the following inputs:

Car Name

Fuel Type

Aspiration

Door Number

Car Body

Drive Wheel

Engine Location

Wheelbase

Car Length, Width, Height

Curb Weight

Engine Type

Cylinder Number

Engine Size

Fuel System

Bore Ratio

Stroke

Compression Ratio

Horsepower

Peak RPM

City MPG

Highway MPG

# Output

Predicted car price displayed instantly on the webpage.

# Key Highlights

End-to-end ML pipeline

Proper feature alignment using saved column metadata

Clean Flask backend

User-friendly frontend

Ready for deployment

Resume & portfolio ready project
