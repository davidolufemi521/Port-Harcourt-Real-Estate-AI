from pathlib import Path

from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd
import os
import numpy as np
app = Flask(__name__)


# This tells the app to look in the exact folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the AI Brain and Features
model = joblib.load(os.path.join(BASE_DIR, "rivers_rent_model.pkl"))
model_features = joblib.load(os.path.join(BASE_DIR, "model_features.pkl"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predictor")
def predictor():
    return render_template("predictor.html")


@app.route("/analysis")
def show_analysis():
    return render_template("analysis.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # 1. Helper to safely grab data or return a SOLID fallback (No NaNs!)
        def get_val(key, fallback, is_numeric=False):
            val = data.get(key)
            if val is None or str(val).strip() == "":
                return fallback
            if is_numeric:
                return float(val)
            return str(val)

        # 2. Build the dictionary with fallback defaults for everything
        input_dict = {}
        for feature in model_features:
            input_dict[feature] = 0  # Ultimate fallback for any random numeric columns
            
        # 3. Safely grab from the web form, using Port Harcourt baselines if they leave it blank
        input_dict["bedrooms"] = get_val("bedrooms", fallback=1, is_numeric=True)
        input_dict["neighborhood"] = get_val("neighborhood", fallback="Other Port Harcourt")
        input_dict["property_type_clean"] = get_val("property_type", fallback="Apartment/Flat")
        input_dict["condition"] = get_val("condition", fallback="Fairly Used")
        input_dict["furnishing"] = get_val("furnishing", fallback="Unfurnished")
        
        # 4. Fill in the hidden columns the web form doesn't ask for!
        if "bathrooms" in input_dict:
            input_dict["bathrooms"] = input_dict["bedrooms"]
        if "toilets" in input_dict:
            input_dict["toilets"] = input_dict["bedrooms"] + 1
        if "size_sqm" in input_dict:
            input_dict["size_sqm"] = input_dict["bedrooms"] * 50

        # 5. Convert to DataFrame. There are ZERO NaNs in this dictionary.
        input_df = pd.DataFrame([input_dict])

        # 6. Make the prediction!
        base_prediction = float(model.predict(input_df)[0])

        # 7. Safety net for implausibly low values
        if base_prediction < 150000:
            base_prediction = 150000 + (int(input_dict["bedrooms"]) * 100000)

        lower_bound = base_prediction * 0.80
        upper_bound = base_prediction * 1.25

        def format_naira(amount):
            return f"\u20A6{int(amount):,}"

        return jsonify(
            {
                "status": "success",
                "exact_estimate": format_naira(base_prediction),
                "range_lower": format_naira(lower_bound),
                "range_upper": format_naira(upper_bound),
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
