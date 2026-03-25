from flask import Flask, request, render_template
import pickle
from pathlib import Path
import numpy as np

# App setup
application = Flask(__name__)
app = application

# Base directory
base = Path(__file__).resolve().parent

# Load models
regressor = pickle.load(open(base / "Models" / "regressor.pkl", "rb"))
scaler = pickle.load(open(base / "Models" / "scaler.pkl", "rb"))

# Home route
@app.route("/")
def index():
    return render_template("index.html")

# Predict route
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # Get form data
            temperature = int(request.form['temperature'])
            rh = int(request.form['rh'])
            dc = float(request.form["dc"])
            rain = float(request.form['rain'])
            ffmc = float(request.form['ffmc'])
            dmc = float(request.form['dmc'])
            isi = float(request.form['isi'])
            bui = float(request.form["bui"])
            classes = int(request.form['classes'])
            region = int(request.form['region'])

            # Arrange data (IMPORTANT: match training order)
            data = np.array([[
                temperature, rh, rain, ffmc, dmc,
                dc, isi, bui, classes, region
            ]])

            # Scale + Predict
            scaled_data = scaler.transform(data)
            prediction = regressor.predict(scaled_data)

            # Clean output
            result = round(prediction[0], 2)

            # Return same page with result
            return render_template("data.html", result=result)

        except Exception as e:
            return render_template("data.html", result=f"Error: {e}")

    # GET request
    return render_template("data.html")


# Run app
if __name__ == "__main__":
    app.run(debug=True)