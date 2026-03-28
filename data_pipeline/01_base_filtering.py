import pandas as pd

# Load the master dataset you just created
print("Loading RIVERS_MASTER_DATASET.csv...\n")
df = pd.read_csv('RIVERS_MASTER_DATASET.csv')

# Count the occurrences of each property type
category_counts = df['property_type'].value_counts()

# Print the results clearly
print("=== ALL PROPERTY CATEGORIES IN YOUR DATASET ===")
print(category_counts)

print("\nTotal Unique Categories:", len(category_counts))


import pandas as pd
import numpy as np

print("Loading Master Dataset...")
df = pd.read_csv('RIVERS_MASTER_DATASET.csv')

# ==========================================
# 1. DROP COMMERCIAL PROPERTIES
# ==========================================
print("Dropping commercial properties...")
commercial_types = [
    'Office Space', 'Warehouse', 'Shop', 'Commercial Property', 
    'Plaza / Complex / Mall', 'Filling Station', 'School', 
    'Conference / Meeting / Training Room'
]
df = df[~df['property_type'].isin(commercial_types)].copy()

# ==========================================
# 2. STANDARDIZE PROPERTY TYPES
# ==========================================
print("Grouping property types...")
def categorize_property(ptype):
    ptype = str(ptype).lower()
    
    if any(x in ptype for x in ['flat', 'apartment', 'penthouse']):
        if 'shared' in ptype or 'studio' in ptype:
            pass 
        else:
            return 'Apartment/Flat'
            
    if any(x in ptype for x in ['duplex', 'terrace', 'townhouse', 'maisonette']):
        return 'Duplex'
        
    if 'bungalow' in ptype:
        return 'Bungalow'
        
    if any(x in ptype for x in ['self contain', 'room', 'bedsitter', 'shared', 'studio']):
        return 'Self Contain / Studio'
        
    if ptype == 'house':
        return 'Detached House'
        
    return 'Other'

df['property_type_clean'] = df['property_type'].apply(categorize_property)

# ==========================================
# 3. STRICTLY KEEP ONLY 'PER ANNUM'
# ==========================================
print("\n--- RENTAL PERIODS BEFORE FILTERING ---")
# Print the exact count of all periods (including missing ones) BEFORE dropping
print(df['period'].value_counts(dropna=False))

print("\nFiltering for STRICTLY 'per annum' listings...")

# Make sure all text is lowercase strings for easy matching
df['period'] = df['period'].astype(str).str.lower()

# Keep ONLY rows where the period contains 'per annum'
df = df[df['period'].str.contains('per annum')].copy()

# ==========================================
# 4. SAVE 
# ==========================================

print(f"\nRemaining Residential Houses (STRICTLY Yearly Rent): {len(df)}")
print("\nNew Clean Categories Distribution:")
print(df['property_type_clean'].value_counts())

df.to_csv("RIVERS_CLEANED_V2.csv", index=False)
print("\n✅ Saved successfully to 'RIVERS_CLEANED_V2.csv'")