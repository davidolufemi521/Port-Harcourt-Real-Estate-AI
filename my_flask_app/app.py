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

        # 1. Start with true mathematical blanks (np.nan) instead of 'None'
        input_df = pd.DataFrame([{feature: np.nan for feature in model_features}])

        # 2. Safely extract data from the web form
        def get_val(key):
            val = data.get(key)
            # If the user left it blank, return np.nan
            if val is None or str(val).strip() == "":
                return np.nan
            return val

        if "bedrooms" in input_df.columns:
            input_df.at[0, "bedrooms"] = get_val("bedrooms")
        if "neighborhood" in input_df.columns:
            input_df.at[0, "neighborhood"] = get_val("neighborhood")
        if "property_type_clean" in input_df.columns:
            input_df.at[0, "property_type_clean"] = get_val("property_type")
        if "condition" in input_df.columns:
            input_df.at[0, "condition"] = get_val("condition")
        if "furnishing" in input_df.columns:
            input_df.at[0, "furnishing"] = get_val("furnishing")

        # 3. Force numeric columns to be numbers, not text!
        numeric_cols = ["bedrooms", "bathrooms", "toilets", "size_sqm"]
        for col in numeric_cols:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

        # 4. Make the prediction (SimpleImputer will now catch the NaNs properly!)
        base_prediction = float(model.predict(input_df)[0])

        # 5. Safety net for implausibly low values
        # We safely get the bedrooms (default to 1 if it was left blank)
        bed_count = input_df.at[0, "bedrooms"]
        if pd.isna(bed_count):
            bed_count = 1 
            
        if base_prediction < 150000:
            base_prediction = 150000 + (int(bed_count) * 100000)

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
