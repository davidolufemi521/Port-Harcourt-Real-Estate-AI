# Port Harcourt Real Estate AI Predictor

## Overview
This project is an end-to-end data science and machine learning system built to estimate fair market rent for residential properties in Port Harcourt.

The workflow covers the full pipeline:
- scraping rental listings across Rivers State
- cleaning and standardizing messy real-world data
- exploring market patterns through EDA
- training machine learning models
- deploying the result in a Flask web application

The final product is a rent prediction tool that returns a practical rent band instead of pretending to know one exact “perfect” price.

---

## Project Goal

The main goal of this project is not just to train a model.

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
- engineering useful features for modeling

This project shows the practical side of data science: most of the real work happens before model training.

---

## The Machine Learning Approach

Several modeling directions were explored for rent estimation.

At first, the project focused on exact price prediction using regression models such as:
- Linear Regression
- Random Forest
- Gradient Boosting

However, due to dataset limitations like:
- small sample size
- duplicated patterns
- inconsistent listing quality
- missing important real-estate features

exact-price prediction was not reliable enough for a user-facing product.

So the deployed version was reframed into a **rent-band prediction system**.

Instead of returning one exact rent figure, the app predicts a more realistic category such as:
- Budget
- Lower Mid
- Upper Mid
- High End
- Luxury

This makes the output more honest, more stable, and more useful for users.

---

## Current Model Output

The web app predicts a rent band and displays a practical rent range.

Current custom rent bands:
- `Budget`: ₦0 - ₦1.5M
- `Lower Mid`: ₦1.5M - ₦3M
- `Upper Mid`: ₦3M - ₦5M
- `High End`: ₦5M - ₦8M
- `Luxury`: ₦8M+

This is more useful than returning a fake-precise number from a small noisy dataset.

---

## Model Performance

Because the dataset is relatively small and imperfect, the goal was not to chase unrealistic accuracy.

The stronger result in practice was a band-based classification model rather than an exact-price regressor.

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
