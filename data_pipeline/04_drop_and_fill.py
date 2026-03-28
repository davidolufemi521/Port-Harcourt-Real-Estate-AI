import pandas as pd

print("Loading the dataset to sweep for physical outliers...\n")
df = pd.read_csv('RIVERS_READY_FOR_ML.csv')

# --- 1. PRE-SWEEP STATS ---
print("=== BEFORE SWEEP ===")
print(f"Bedrooms Skewness: {df['bedrooms'].skew():.2f}")
print(f"Size Skewness: {df['size_sqm'].skew():.2f}")
starting_rows = len(df)

# --- 2. THE CHOPPING BLOCK ---
# Filter out anything with more than 6 bedrooms
df_cleaned = df[df['bedrooms'] <= 6]

# Filter out anything larger than 1500 sqm
df_cleaned = df_cleaned[df_cleaned['size_sqm'] <= 1500]

# --- 3. POST-SWEEP STATS ---
ending_rows = len(df_cleaned)
dropped_rows = starting_rows - ending_rows

print(f"\n🚨 Dropped {dropped_rows} massive outlier properties.")

print("\n=== AFTER SWEEP ===")
print(f"New Bedrooms Skewness: {df_cleaned['bedrooms'].skew():.2f}")
print(f"New Size Skewness: {df_cleaned['size_sqm'].skew():.2f}")
print("===================\n")

# --- 4. SAVE THE TRULY FINAL DATASET ---
# Overwriting the file so it's perfectly clean for Phase 2
df_cleaned.to_csv('RIVERS_READY_FOR_ML.csv', index=False)
print("💾 Overwrote 'RIVERS_READY_FOR_ML.csv' with the perfectly pruned data.")