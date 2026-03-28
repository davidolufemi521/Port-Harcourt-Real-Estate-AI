import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor

print("Loading data to build the AI Brain...\n")
df = pd.read_csv('RIVERS_ML_ENCODED.csv')

# Drop 'period' just in case
if 'period' in df.columns:
    df = df.drop(columns=['period'])

# Separate the answers from the test
X = df.drop(columns=['price'])
y = df['price']

print("🧠 Training the Champion (Gradient Boosting) on the ENTIRE dataset...")
# We use Gradient Boosting because it had the lowest real-world Naira error (MAE)
final_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
final_model.fit(X, y)

print("💾 Saving the AI Brain files to your computer...")
# Export the model and the exact columns it expects to see
joblib.dump(final_model, 'rivers_rent_model.pkl')
joblib.dump(list(X.columns), 'model_features.pkl')

print("✅ SUCCESS! The files 'rivers_rent_model.pkl' and 'model_features.pkl' have been created.")
print("You can now start your web app and calculate the rent!")