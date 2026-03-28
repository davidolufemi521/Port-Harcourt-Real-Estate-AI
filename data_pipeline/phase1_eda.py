import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create a folder to save our graphs for the final report
output_dir = 'eda_graphs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Loading RIVERS_CLEANED_FINAL.csv...\n")
df = pd.read_csv('RIVERS_CLEANED_FINAL.csv')

# Set a professional visual theme for our charts
sns.set_theme(style="whitegrid", palette="muted")

print("--- Starting Phase 1: Univariate Analysis ---\n")

# ==========================================
# PART A: CATEGORICAL DATA (Bar Charts)
# ==========================================
print("Generating Categorical Charts...")

# 1. Top 15 Neighborhoods
plt.figure(figsize=(10, 6))
top_neighborhoods = df['neighborhood'].value_counts().nlargest(15)
sns.barplot(y=top_neighborhoods.index, x=top_neighborhoods.values, hue=top_neighborhoods.index, legend=False)
plt.title('Top 15 Most Active Neighborhoods for Rent')
plt.xlabel('Number of Properties')
plt.ylabel('Neighborhood')
plt.tight_layout()
plt.savefig(f'{output_dir}/1_top_neighborhoods.png', dpi=300)
plt.close()

# 2. Property Condition & Furnishing (Side-by-side)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, x='condition', hue='condition', ax=axes[0], legend=False)
axes[0].set_title('Distribution of Property Condition')
axes[0].tick_params(axis='x', rotation=45)

sns.countplot(data=df, x='furnishing', hue='furnishing', ax=axes[1], legend=False)
axes[1].set_title('Distribution of Furnishing Status')
plt.tight_layout()
plt.savefig(f'{output_dir}/2_condition_and_furnishing.png', dpi=300)
plt.close()

# 3. Property Types
plt.figure(figsize=(10, 5))
sns.countplot(data=df, y='property_type_clean', hue='property_type_clean', order=df['property_type_clean'].value_counts().index, legend=False)
plt.title('Breakdown of Property Types')
plt.xlabel('Count')
plt.ylabel('Property Type')
plt.tight_layout()
plt.savefig(f'{output_dir}/3_property_types.png', dpi=300)
plt.close()

# ==========================================
# PART B: NUMERICAL DATA & OUTLIERS
# ==========================================
print("Generating Numerical Distributions and Boxplots...")

# 4. Histograms for Bedrooms, Bathrooms, Toilets
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df['bedrooms'], bins=10, kde=True, ax=axes[0], color='blue')
axes[0].set_title('Bedrooms Distribution')

sns.histplot(df['bathrooms'], bins=10, kde=True, ax=axes[1], color='green')
axes[1].set_title('Bathrooms Distribution')

sns.histplot(df['toilets'], bins=10, kde=True, ax=axes[2], color='red')
axes[2].set_title('Toilets Distribution')
plt.tight_layout()
plt.savefig(f'{output_dir}/4_room_distributions.png', dpi=300)
plt.close()

# 5. Outlier Detection (Boxplots)
plt.figure(figsize=(10, 4))
sns.boxplot(data=df[['bedrooms', 'bathrooms', 'toilets']], orient="h")
plt.title('Outlier Check: Room Counts')
plt.tight_layout()
plt.savefig(f'{output_dir}/5_room_outliers_boxplot.png', dpi=300)
plt.close()

# 6. Size (Sqm) Distribution and Outliers
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.histplot(df['size_sqm'], bins=30, kde=True, ax=axes[0], color='purple')
axes[0].set_title('Property Size (Sqm) Distribution')

sns.boxplot(x=df['size_sqm'], ax=axes[1], color='purple')
axes[1].set_title('Outlier Check: Property Size')
plt.tight_layout()
plt.savefig(f'{output_dir}/6_size_distribution_and_outliers.png', dpi=300)
plt.close()

# ==========================================
# PART C: SKEWNESS REPORT
# ==========================================
print("\n=== MATHEMATICAL SKEWNESS REPORT ===")
# Skewness between -1 and 1 is highly symmetrical. 
# Anything > 1 or < -1 is highly skewed.
numerical_cols = ['bedrooms', 'bathrooms', 'toilets', 'size_sqm']
print(df[numerical_cols].skew())
print("====================================")

print(f"\n✅ Phase 1 Complete! Check the '{output_dir}' folder in your project directory to see your graphs.")