import pandas as pd

print("Loading RIVERS_CLEANED_V4.csv...\n")
df = pd.read_csv('RIVERS_CLEANED_V4.csv')

# --- 1. Handle the Remaining Missing Data ---
df['furnishing'] = df['furnishing'].fillna('Not Specified')
df['condition'] = df['condition'].fillna('Not Specified')
print("✅ Filled remaining missing furnishing and condition with 'Not Specified'.")

# --- 2. Drop the Unwanted Columns ---
columns_to_drop = ['property_type', 'amenities', 'description']

# We check if they exist first just to prevent any spelling/Key errors
actual_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
df = df.drop(columns=actual_cols_to_drop)
print(f"✅ Dropped columns: {actual_cols_to_drop}\n")

# --- 3. Inspect the Dataset for Any Remaining Missing Data ---
print("================ DATASET INFO ================")
# .info() gives us the column types and non-null counts
df.info()
print("==============================================\n")

print("=== EXACT MISSING VALUES COUNT PER COLUMN ===")
# .isna().sum() gives us a direct count of missing values per column
print(df.isna().sum())
print("=============================================\n")

# --- 4. Save the Final Version ---
df.to_csv('RIVERS_CLEANED_V5.csv', index=False)
print("💾 Saved successfully as RIVERS_CLEANED_V5.csv")