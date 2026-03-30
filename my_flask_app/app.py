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

        def get_val(key, fallback, is_numeric=False):
            val = data.get(key)
            if val is None or str(val).strip() == "":
                return fallback
            if is_numeric:
                return float(val)
            return str(val)

        # 1. Create a blank row with EXACTLY the encoded columns the model knows (all 0s)
        input_dict = {feature: 0 for feature in model_features}
            
        # 2. Fill in the numbers
        bedrooms = get_val("bedrooms", fallback=1, is_numeric=True)
        if "bedrooms" in input_dict: input_dict["bedrooms"] = bedrooms
        if "bathrooms" in input_dict: input_dict["bathrooms"] = bedrooms
        if "toilets" in input_dict: input_dict["toilets"] = bedrooms + 1
        if "size_sqm" in input_dict: input_dict["size_sqm"] = bedrooms * 50

        # 3. THE MAGIC FIX: Flip the switch (0 to 1) for the correct encoded category!
        neighborhood = get_val("neighborhood", fallback="Other Port Harcourt")
        if f"neighborhood_{neighborhood}" in input_dict:
            input_dict[f"neighborhood_{neighborhood}"] = 1

        property_type = get_val("property_type", fallback="Apartment/Flat")
        if f"property_type_clean_{property_type}" in input_dict:
            input_dict[f"property_type_clean_{property_type}"] = 1
        elif f"property_type_{property_type}" in input_dict:
            input_dict[f"property_type_{property_type}"] = 1

        condition = get_val("condition", fallback="Fairly Used")
        if f"condition_{condition}" in input_dict:
            input_dict[f"condition_{condition}"] = 1

        furnishing = get_val("furnishing", fallback="Unfurnished")
        if f"furnishing_{furnishing}" in input_dict:
            input_dict[f"furnishing_{furnishing}"] = 1

        # 4. Convert to DataFrame (ZERO unseen columns now!)
        input_df = pd.DataFrame([input_dict])

        # 5. Make the prediction!
        base_prediction = float(model.predict(input_df)[0])

        if base_prediction < 150000:
            base_prediction = 150000 + (int(bedrooms) * 100000)

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
