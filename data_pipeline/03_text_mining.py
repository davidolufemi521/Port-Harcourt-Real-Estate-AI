import pandas as pd
import numpy as np
import re

print("Loading V3 Dataset...\n")
df = pd.read_csv('RIVERS_CLEANED_V3.csv')

# --- 1. TITLE RESCUE FUNCTIONS ---
def rescue_furnishing_title(row):
    if pd.notna(row['furnishing']):
        return row['furnishing']
    title = str(row['title']).lower()
    if 'furnished' in title or 'furnish' in title:
        if 'semi' in title or 'half' in title:
            return 'Semi-Furnished'
        return 'Furnished'
    return np.nan

def rescue_condition_title(row):
    if pd.notna(row['condition']):
        return row['condition']
    title = str(row['title']).lower()
    if any(word in title for word in ['new', 'brand new', 'virgin']):
        return 'Newly-Built'
    if 'renovated' in title:
        return 'Renovated'
    return np.nan

# --- 2. DESCRIPTION RESCUE FUNCTIONS ---
def extract_furnishing_desc(text):
    if pd.isna(text): return None
    text = str(text).lower()
    if re.search(r'\bunfurnished\b|\bnot furnished\b', text): return 'Unfurnished'
    if re.search(r'\bfurnished\b|\bbed\b|\bsofa\b|\btelevision\b|\btv\b|\bfurniture\b', text): return 'Furnished'
    if re.search(r'\b(?:semi|partly|half)[\s-]?furnished\b', text): return 'Semi-Furnished'
    return None

def extract_condition_desc(text):
    if pd.isna(text): return None
    text = str(text).lower()
    if re.search(r'\b(?:newly built|brand new|new|just built|virgin)\b', text): return 'Newly-Built'
    if re.search(r'\b(?:renovated|refurbished|newly painted)\b', text): return 'Renovated'
    if re.search(r'\b(?:fairly used|used|old|needs work)\b', text): return 'Fairly Used'
    if re.search(r'\b(?:uncompleted|carcass|under construction)\b', text): return 'Uncompleted'
    return None

# --- 3. EXECUTE THE RESCUES ---
print("Phase 1: Hunting for clues in titles...")
df['furnishing'] = df.apply(rescue_furnishing_title, axis=1)
df['condition'] = df.apply(rescue_condition_title, axis=1)

print("Phase 2: Deep diving into descriptions...")
# Only apply description rescue to rows that are STILL missing data after the title rescue
missing_furn = df['furnishing'].isna()
df.loc[missing_furn, 'furnishing'] = df.loc[missing_furn, 'description'].apply(extract_furnishing_desc)

missing_cond = df['condition'].isna()
df.loc[missing_cond, 'condition'] = df.loc[missing_cond, 'description'].apply(extract_condition_desc)

# --- 4. PRINT FINAL RESULTS ---
print("\n=== FINAL RESCUE RESULTS ===")
print("FURNISHING STILL MISSING:", df['furnishing'].isna().sum())
print("CONDITION STILL MISSING:", df['condition'].isna().sum())

# --- 5. SAVE THE DATA (THE MOST IMPORTANT STEP) ---
print("\nSaving results to RIVERS_CLEANED_V4.csv...")
df.to_csv('RIVERS_CLEANED_V4.csv', index=False)
print("Save complete! Your data is officially rescued.")