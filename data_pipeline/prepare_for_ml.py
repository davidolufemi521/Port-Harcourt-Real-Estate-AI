import pandas as pd

print("Loading the dataset for Final ML Prep...\n")
df = pd.read_csv('RIVERS_READY_FOR_ML.csv')

# --- 1. DROP THE NOISE ---
df = df.drop(columns=['bathrooms', 'toilets', 'size_sqm', 'source', 'url', 'title'])

# --- 2. ENGINEER THE SMART FLAGS ---
# The Luxury Area Flag
luxury_areas = ['GRA / Eagle Island', 'Peter Odili / Trans Amadi / Abuloma', 'Eliouzu / Rumuodara / Rumuibekwe']
df['is_luxury_area'] = df['neighborhood'].apply(lambda x: 1 if x in luxury_areas else 0)

# The Virgin Flag
df['is_newly_built'] = df['condition'].apply(lambda x: 1 if x == 'Newly-Built' else 0)

# The Furnished Flag
df['is_furnished'] = df['furnishing'].apply(lambda x: 1 if x == 'Furnished' else 0)

# --- 3. TRANSLATE THE REST TO MATH (One-Hot Encoding) ---
# This turns the remaining text categories (like Property Type) into 1s and 0s
df_encoded = pd.get_dummies(df, columns=['property_type_clean', 'neighborhood', 'condition', 'furnishing'], drop_first=True)

# Ensure all boolean columns (True/False) from get_dummies are converted to 1/0 integers
for col in df_encoded.columns:
    if df_encoded[col].dtype == bool:
        df_encoded[col] = df_encoded[col].astype(int)

print("=== AI READY DATASET ===")
print(f"Total columns for AI to learn from: {len(df_encoded.columns)}")
print("========================\n")

# --- 4. SAVE THE ULTIMATE FILE ---
df_encoded.to_csv('RIVERS_ML_ENCODED.csv', index=False)
print("💾 Saved as 'RIVERS_ML_ENCODED.csv'.")