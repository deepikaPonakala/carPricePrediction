from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model and feature columns
rf_model = pickle.load(open("models/rf_model.pkl", "rb"))
feature_columns = pickle.load(open("models/feature_columns.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    input_data = {
        "CarName": request.form["CarName"],
        "fueltype": request.form["fueltype"],
        "aspiration": request.form["aspiration"],
        "doornumber": request.form["doornumber"],
        "carbody": request.form["carbody"],
        "drivewheel": request.form["drivewheel"],
        "enginelocation": request.form["enginelocation"],
        "wheelbase": float(request.form["wheelbase"]),
        "carlength": float(request.form["carlength"]),
        "carwidth": float(request.form["carwidth"]),
        "carheight": float(request.form["carheight"]),
        "curbweight": float(request.form["curbweight"]),
        "enginetype": request.form["enginetype"],
        "cylindernumber": request.form["cylindernumber"],
        "enginesize": float(request.form["enginesize"]),
        "fuelsystem": request.form["fuelsystem"],
        "boreratio": float(request.form["boreratio"]),
        "stroke": float(request.form["stroke"]),
        "compressionratio": float(request.form["compressionratio"]),
        "horsepower": float(request.form["horsepower"]),
        "peakrpm": float(request.form["peakrpm"]),
        "citympg": float(request.form["citympg"]),
        "highwaympg": float(request.form["highwaympg"])
    }

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # One-hot encode (SAME AS TRAINING)
    df = pd.get_dummies(df)

    # Align with training columns
    df = df.reindex(columns=feature_columns, fill_value=0)

    prediction = round(rf_model.predict(df)[0], 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
