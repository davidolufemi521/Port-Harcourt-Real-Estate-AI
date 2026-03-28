import pandas as pd
import numpy as np

print("Loading Cleaned Dataset V2...")
df = pd.read_csv('RIVERS_CLEANED_V2.csv')

def extract_neighborhood(row):
    text_to_search = str(row['property_address']).lower() + " " + str(row['title']).lower()
    
    # Tier 1: Luxury / Highbrow
    if any(word in text_to_search for word in ['gra', 'g.r.a', 'tombia', 'abacha', 'eagle island']):
        return 'GRA / Eagle Island'
    if any(word in text_to_search for word in ['peter odili', 'trans amadi', 'odili', 'golf', 'abuloma']):
        return 'Peter Odili / Trans Amadi / Abuloma'
        
    # Tier 2: Mid-Level / Popular Residential
    if any(word in text_to_search for word in ['woji', 'alcon']):
        return 'Woji'
    if any(word in text_to_search for word in ['stadium', 'rumuomasi', 'olu obasanjo', 'waterline']):
        return 'Stadium / Rumuomasi / Olu Obasanjo'
    if any(word in text_to_search for word in ['ada george', 'adageorge', 'agip', 'mgbuoba', 'nta', 'eliopranwo', 'elioparanwo']):
        return 'Ada George / Agip / NTA'
    if any(word in text_to_search for word in ['eliozu', 'okporo', 'artillery', 'rumuodara', 'rumudara', 'aba road', 'rumuibekwe', 'rumuebekwe', 'rumuogba', 'cocaine']):
        return 'Eliozu / Rumuodara / Rumuibekwe'
    if any(word in text_to_search for word in ['rumuola', 'rumuigbo', 'rumuokwuta', 'rumuokoro', 'psychiatric', 'rumuodomaya']):
        return 'Rumuola / Rumuokoro / Rumuodomaya'
        
    # Tier 3: Developing / Outskirts / Specific Hubs
    if any(word in text_to_search for word in ['eneka', 'rumuduru', 'rumunduru', 'elimgbu', 'elimbu', 'rumuokwurusi', 'rumukurushi', 'rumuokurushi', 'atali', 'shell coop', 'shell corp', 'shell corperative', 'tank', 'apu road']):
        return 'Eneka / Rumuduru / Rumuokwurusi'
    if any(word in text_to_search for word in ['sars', 'rumuagholu', 'rumarholu', 'rumuogholu', 'rukpakulusi', 'naf harmony', 'nafharmony', 'naf estate', 'g.u ake']):
        return 'SARS Rd / Rumuagholu / NAF Estate'
    if any(word in text_to_search for word in ['choba', 'ozuoba', 'uniport', 'alakahia']):
        return 'Choba / Ozuoba'
    if any(word in text_to_search for word in ['elelenwo', 'akpajo', 'eleme', 'oyigbo']):
        return 'Elelenwo / Eleme'
    if any(word in text_to_search for word in ['igwuruta', 'rukpokwu', 'rukporkwu', 'rukpuokwu', 'airport', 'ikwerre', 'elikporpodu']):
        return 'Igwuruta / Rukpokwu'
    if any(word in text_to_search for word in ['iwofe', 'orazi', 'egbelu']):
        return 'Iwofe / Orazi'
        
    # Tier 4: General Port Harcourt
    if any(word in text_to_search for word in ['port harcourt', 'rivers', 'phc', 'portharcourt', 'east west']):
        return 'Other Port Harcourt'
        
    # THE TRAPDOOR: Real Out of State Houses
    return 'OUT_OF_STATE'

print("Extracting clean neighborhoods and organizing Port Harcourt zones...")
df['neighborhood'] = df.apply(extract_neighborhood, axis=1)

# --- 1. PRINT OUT OF STATE LEAKS ---
out_of_state_df = df[df['neighborhood'] == 'OUT_OF_STATE']
print(f"\nCaught {len(out_of_state_df)} out-of-state properties (Lagos, Abuja, etc.).")

# --- 2. DROP THE ACTUAL OUT OF STATE HOUSES ---
df = df[df['neighborhood'] != 'OUT_OF_STATE'].copy()

# --- 3. PRINT FINAL COUNTS AND SAVE ---
print("\n--- FINAL NEIGHBORHOOD ZONES ---")
print(df['neighborhood'].value_counts())

# Drop the old messy address column, we don't need it anymore!
df = df.drop(columns=['property_address'])

df.to_csv("RIVERS_CLEANED_V3.csv", index=False)
print(f"\n✅ Total Houses Ready for Machine Learning: {len(df)}")
print("✅ Saved successfully to 'RIVERS_CLEANED_V3.csv'")