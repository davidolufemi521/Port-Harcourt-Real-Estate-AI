import pandas as pd
import numpy as np
import re

print("Loading datasets...")
# Load the two files you scraped
df_jiji = pd.read_csv('rivers_rentals_hybrid.csv')
df_npc = pd.read_csv('rivers_properties_updated.csv')

# ==========================================
# 1. PREPARE NIGERIA PROPERTY CENTRE DATA
# ==========================================
print("Cleaning Nigeria Property Centre data...")

# Map NPC columns to match Jiji's column names
npc_mapping = {
    'URL': 'url',
    'Title': 'title',
    'Price': 'price',
    'Period': 'period',
    'Type': 'property_type',
    'Bedrooms': 'bedrooms',
    'Bathrooms': 'bathrooms',
    'Toilets': 'toilets',
    'Location': 'property_address',
    'Furnishing': 'furnishing',
    'Total Area': 'size_sqm',
    'Description': 'description'
}
df_npc = df_npc.rename(columns=npc_mapping)

# Clean NPC Price (Remove commas and convert text to numbers)
df_npc['price'] = df_npc['price'].astype(str).str.replace(',', '', regex=False)
df_npc['price'] = df_npc['price'].str.replace(r'[^\d.]', '', regex=True) # Remove Naira signs
df_npc['price'] = pd.to_numeric(df_npc['price'], errors='coerce')

# Clean NPC Period (make everything lowercase so it matches Jiji)
df_npc['period'] = df_npc['period'].astype(str).str.strip().str.lower()

# Clean NPC Size (Remove ' sqm' and convert to number)
df_npc['size_sqm'] = df_npc['size_sqm'].astype(str).str.replace(' sqm', '', regex=False)
df_npc['size_sqm'] = df_npc['size_sqm'].str.replace(',', '', regex=False)
df_npc['size_sqm'] = pd.to_numeric(df_npc['size_sqm'], errors='coerce')

# Add missing columns so it matches Jiji perfectly
df_npc['amenities'] = np.nan
df_npc['condition'] = np.nan
df_npc['source'] = 'Nigeria Property Centre' # Track where it came from


# ==========================================
# 2. PREPARE JIJI DATA
# ==========================================
print("Cleaning Jiji data...")

# Rename price column to match
df_jiji = df_jiji.rename(columns={'price_cleaned': 'price'})

# Clean Jiji Period
df_jiji['period'] = df_jiji['period'].astype(str).str.strip().str.lower()

# Add missing columns so it matches NPC
df_jiji['description'] = np.nan
df_jiji['source'] = 'Jiji'


# ==========================================
# 3. MERGE THEM TOGETHER
# ==========================================
print("Merging the datasets...")

# These are the "Gold Standard" columns we want to keep for Machine Learning
master_columns = [
    'source', 'url', 'title', 'price', 'period', 'property_type', 
    'bedrooms', 'bathrooms', 'toilets', 'property_address', 
    'size_sqm', 'furnishing', 'condition', 'amenities', 'description'
]

# Select only the master columns
df_npc_clean = df_npc[[c for c in master_columns if c in df_npc.columns]].copy()
df_jiji_clean = df_jiji[[c for c in master_columns if c in df_jiji.columns]].copy()

# Stack them vertically!
df_master = pd.concat([df_jiji_clean, df_npc_clean], ignore_index=True)

# Remove any completely identical URLs to prevent duplicates
initial_count = len(df_master)
df_master = df_master.drop_duplicates(subset=['url'])
final_count = len(df_master)

print(f"Removed {initial_count - final_count} duplicate URLs.")
print(f"Final Dataset Size: {final_count} Unique Houses!")

# Save to a brand new, clean CSV
df_master.to_csv("RIVERS_MASTER_DATASET.csv", index=False)
print("✅ Saved to 'RIVERS_MASTER_DATASET.csv'. Your data is ready!")