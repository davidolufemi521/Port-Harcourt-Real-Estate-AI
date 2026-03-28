# Port Harcourt Real Estate AI Predictor

## Overview
This project is an end-to-end data science and machine learning system built to estimate fair market rent for residential properties in Port Harcourt.

The workflow covers the full pipeline:
- scraping rental listings across Rivers State
- cleaning and standardizing messy real-world data
- exploring market patterns through EDA
- training machine learning models
- deploying the result in a Flask web application

The final product is a rent prediction tool that returns an estimated rent together with a practical market range.

---

## Project Goal

The goal of this project is not just to train a model.

It is to take unreliable property listing data and turn it into a useful decision-support tool for:
- renters
- landlords
- investors
- anyone trying to understand the Port Harcourt rental market

Because real estate listings are noisy and incomplete, the project is designed as a **fair market rent estimator**, not a perfect valuation engine.

---

## The Data Pipeline

Real-world property data is messy, and this project reflects that clearly.

The repository includes:
- scraped rental listings from Rivers State
- cleaned datasets at different stages of processing
- machine-learning-ready datasets
- scripts for cleaning, transformation, feature preparation, and training

The data preparation process included:
- removing irrelevant and inconsistent records
- handling missing values
- standardizing categories such as neighborhood, furnishing, and property condition
- removing duplicate listings
- preparing raw features for machine learning

One important improvement in the final version was removing duplicate listings before model training, which helped make the model more reliable.

This project shows the practical side of data science: most of the real work happens before model training.

---

## The Machine Learning Approach

Several modeling directions were explored for rent estimation, including:
- Linear Regression
- Random Forest
- Gradient Boosting

The final deployed version uses a **Gradient Boosting Regressor** trained on the cleaned raw dataset through a preprocessing pipeline.

The model uses:
- numeric features such as bedrooms, bathrooms, toilets, and size
- categorical features such as neighborhood, furnishing, property type, and condition
- automatic preprocessing with imputing and one-hot encoding inside the pipeline

Instead of showing only one exact predicted value, the web app presents:
- a central estimated rent
- a practical fair market range around that estimate

This makes the output more realistic and safer for user decision-making.

---

## Current Model Output

The web app predicts:
- an estimated annual rent
- a lower market bound
- an upper market bound

The displayed range is based on the model prediction:
- lower bound: about `-20%`
- upper bound: about `+25%`

This helps users treat the result as a negotiation and market guidance tool, not as a perfectly exact valuation.

---

## Model Performance

After cleaning the data and removing duplicates, the model achieved approximately:

- Holdout MAE: `₦1,800,525.51`
- Holdout R²: `0.6105`
- 5-Fold Cross-Validation MAE: `₦1,704,828.83`
- 5-Fold Cross-Validation R²: `0.5874`

These results are not perfect, but they are realistic for a small real-world property dataset with noisy listings.

### Why the Model Has Limits

Important real-world pricing factors are missing from many listings, such as:
- road quality
- electricity reliability
- drainage and flood risk
- compound quality
- security level
- exact micro-location
- agent markup or inconsistent pricing
- unreliable size estimates

That means the model should be treated as a **market baseline tool**, not a replacement for human property valuation.

---

## Key Insights From the Analysis

Some important patterns emerged during the project:

- **Newly built properties command a premium**
  Newly built houses and flats tend to rent for more than older or fairly used ones.

- **Location matters heavily**
  Neighborhood grouping had a strong effect on rent levels, especially in more premium areas.

- **Bedrooms were more reliable than size**
  Bedroom count was a more dependable pricing signal than square meter values, which were often noisy or inconsistently reported.

- **Data quality affects model quality**
  Duplicate listings, vague descriptions, and inconsistent formatting had a direct impact on prediction performance.

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Flask
- HTML/CSS
- Joblib

---

## Running the Project

1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
