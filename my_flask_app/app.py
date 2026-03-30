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

        # Build one raw input row and let the saved sklearn pipeline preprocess it.
        input_df = pd.DataFrame(
            [{feature: None for feature in model_features}],
            dtype=object,
        )

        if "bedrooms" in input_df.columns:
            input_df.at[0, "bedrooms"] = int(data["bedrooms"])
        if "neighborhood" in input_df.columns:
            input_df.at[0, "neighborhood"] = data.get("neighborhood")
        if "property_type_clean" in input_df.columns:
            input_df.at[0, "property_type_clean"] = data.get("property_type")
        if "condition" in input_df.columns:
            input_df.at[0, "condition"] = data.get("condition")
        if "furnishing" in input_df.columns:
            input_df.at[0, "furnishing"] = data.get("furnishing")
        
        input_df = input_df.replace(r'^\s*$', np.nan, regex=True)



        base_prediction = float(model.predict(input_df)[0])

        # Safety net for implausibly low values.
        if base_prediction < 150000:
            base_prediction = 150000 + (int(data["bedrooms"]) * 100000)

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


if __name__ == "__main__":
    app.run(debug=True)
