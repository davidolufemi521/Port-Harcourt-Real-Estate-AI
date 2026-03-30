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

        # 1. Helper to safely extract and convert data from the web form
        def get_val(key, is_numeric=False):
            val = data.get(key)
            if val is None or str(val).strip() == "":
                return np.nan
            if is_numeric:
                return float(val) # Forces the text '4' to become a math 4.0
            return str(val)

        # 2. Build the data as a simple dictionary first (no strict Pandas rules yet)
        input_dict = {feature: np.nan for feature in model_features}

        # 3. Fill the dictionary
        if "bedrooms" in input_dict:
            input_dict["bedrooms"] = get_val("bedrooms", is_numeric=True)
        if "neighborhood" in input_dict:
            input_dict["neighborhood"] = get_val("neighborhood")
        if "property_type_clean" in input_dict:
            input_dict["property_type_clean"] = get_val("property_type")
        if "condition" in input_dict:
            input_dict["condition"] = get_val("condition")
        if "furnishing" in input_dict:
            input_dict["furnishing"] = get_val("furnishing")

        # 4. NOW convert it to a DataFrame. Pandas will automatically handle the types!
        input_df = pd.DataFrame([input_dict])

        # 5. Make the prediction
        base_prediction = float(model.predict(input_df)[0])

        # 6. Safety net for implausibly low values
        bed_count = input_dict.get("bedrooms")
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
