import pandas as pd

print("Loading Cleaned Dataset V2 (because it still has the addresses!)...")
df = pd.read_csv('RIVERS_CLEANED_V2.csv')

# We use our finalized V3 rules here just to check the data
def extract_neighborhood(row):
    text_to_search = str(row['property_address']).lower() + " " + str(row['title']).lower()
    
    # Tier 1
    if any(word in text_to_search for word in ['gra', 'g.r.a', 'tombia', 'abacha', 'eagle island']):
        return 'GRA / Eagle Island'
    if any(word in text_to_search for word in ['peter odili', 'trans amadi', 'odili', 'golf']):
        return 'Peter Odili / Trans Amadi'
        
    # Tier 2
    if any(word in text_to_search for word in ['woji', 'alcon']):
        return 'Woji'
    if any(word in text_to_search for word in ['stadium', 'rumuomasi', 'olu obasanjo', 'waterline']):
        return 'Stadium / Rumuomasi / Olu Obasanjo'
    if any(word in text_to_search for word in ['ada george', 'adageorge', 'agip', 'mgbuoba', 'nta', 'eliopranwo', 'elioparanwo']):
        return 'Ada George / Agip / NTA'
    if any(word in text_to_search for word in ['eliozu', 'okporo', 'artillery', 'rumuodara', 'rumudara', 'aba road', 'rumuibekwe', 'rumuebekwe', 'rumuogba', 'cocaine']):
        return 'Eliozu / Rumuodara / Rumuibekwe'
    if any(word in text_to_search for word in ['rumuola', 'rumuigbo', 'rumuokwuta', 'rumuokoro', 'psychiatric']):
        return 'Rumuola / Rumuokoro'
        
    # Tier 3
    if any(word in text_to_search for word in ['eneka', 'rumuduru', 'rumunduru', 'elimgbu', 'elimbu', 'rumuokwurusi', 'rumukurushi', 'rumuokurushi', 'atali', 'shell coop', 'shell corp', 'shell corperative', 'tank']):
        return 'Eneka / Rumuduru / Rumuokwurusi'
    if any(word in text_to_search for word in ['sars', 'rumuagholu', 'rumarholu', 'rumuogholu', 'rukpakulusi', 'naf harmony', 'naf estate', 'g.u ake']):
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
    if any(word in text_to_search for word in ['port harcourt', 'rivers', 'phc', 'portharcourt', 'abuloma', 'east west']):
        return 'Other Port Harcourt'
        
    return 'OUT_OF_STATE'

df['neighborhood'] = df.apply(extract_neighborhood, axis=1)

# Filter for only the 'Other' bucket
other_ph = df[df['neighborhood'] == 'Other Port Harcourt']

# Get the unique addresses and drop blanks
unique_addresses = other_ph['property_address'].dropna().unique()

print(f"\nTotal properties in 'Other Port Harcourt': {len(other_ph)}")
print(f"Total UNIQUE addresses to review: {len(unique_addresses)}\n")
print("=== ADDRESSES TO REVIEW ===")

for address in unique_addresses:
    print(f"- {address}")