import pickle
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load model and encoders
rf_model = pickle.load(open("models/rf_model.pkl", "rb"))
encoders = pickle.load(open("models/encoders.pkl", "rb"))

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get form values
        car_data = {
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
            "curbweight": int(request.form["curbweight"]),
            "enginetype": request.form["enginetype"],
            "cylindernumber": request.form["cylindernumber"],
            "enginesize": int(request.form["enginesize"]),
            "fuelsystem": request.form["fuelsystem"],
            "boreratio": float(request.form["boreratio"]),
            "stroke": float(request.form["stroke"]),
            "compressionratio": float(request.form["compressionratio"]),
            "horsepower": int(request.form["horsepower"]),
            "peakrpm": int(request.form["peakrpm"]),
            "citympg": int(request.form["citympg"]),
            "highwaympg": int(request.form["highwaympg"]),
        }

        # Convert to dataframe
        df_input = pd.DataFrame([car_data])

        # Encode categorical features
        for col, le in encoders.items():
            if col in df_input.columns:
                if df_input[col].iloc[0] in le.classes_:
                    df_input[col] = le.transform(df_input[col])
                else:
                    df_input[col] = -1

        # Prediction
        prediction = rf_model.predict(df_input)[0]

        return render_template("index.html", prediction_text=f"Predicted Car Price: ${prediction:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)