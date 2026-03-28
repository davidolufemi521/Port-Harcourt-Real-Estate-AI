import pandas as pd

print("Loading RIVERS_CLEANED_V5.csv...\n")
# This is the line we were missing! We have to load the data into 'df' first.
df = pd.read_csv('RIVERS_CLEANED_V5.csv')

print("--- Starting Final Polish ---")

# 1. Drop the tiny missing rows for bedrooms and bathrooms
df = df.dropna(subset=['bedrooms', 'bathrooms'])
print("✅ Dropped rows with missing bedrooms or bathrooms.")

# 2. Smart fill for Toilets (Fill with the number of bathrooms)
df['toilets'] = df['toilets'].fillna(df['bathrooms'])
print("✅ Filled missing toilets with bathroom counts.")

# 3. Smart fill for Size (Fill with the median size of houses with the SAME number of bedrooms)
df['size_sqm'] = df.groupby('bedrooms')['size_sqm'].transform(lambda x: x.fillna(x.median()))

# Just in case a unique bedroom count had no median to pull from, fill any leftovers with the overall median
df['size_sqm'] = df['size_sqm'].fillna(df['size_sqm'].median())
print("✅ Filled missing sizes using median bedroom sizes.")

print("\n=== THE FINAL MISSING VALUES CHECK ===")
print(df.isna().sum())

# Save the ultimate clean version!
df.to_csv('RIVERS_CLEANED_FINAL.csv', index=False)
print("\n💾 Saved successfully as RIVERS_CLEANED_FINAL.csv")