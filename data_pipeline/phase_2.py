import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Loading the perfectly cleaned ML dataset...\n")
df = pd.read_csv('RIVERS_READY_FOR_ML.csv')

sns.set_theme(style="whitegrid", palette="muted")
output_dir = 'eda_graphs'

print("--- Starting Phase 2: Bivariate Analysis ---\n")

# ==========================================
# 1. Price vs. Location (Median Rent)
# ==========================================
print("Generating Location Analysis...")
plt.figure(figsize=(12, 6))
# We use Median instead of Mean because it's more accurate for prices
location_prices = df.groupby('neighborhood')['price'].median().sort_values(ascending=False)

sns.barplot(x=location_prices.values, y=location_prices.index, hue=location_prices.index, legend=False, palette="viridis")
plt.title('Median Rent Price by Neighborhood in Port Harcourt')
plt.xlabel('Median Price (Naira)')
plt.ylabel('Neighborhood')
plt.ticklabel_format(style='plain', axis='x')
plt.tight_layout()
plt.savefig(f'{output_dir}/8_price_by_location.png', dpi=300)
plt.close()

# ==========================================
# 2. Price vs. Property Type
# ==========================================
print("Generating Property Type Analysis...")
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x='price', y='property_type_clean', hue='property_type_clean', legend=False, palette="Set2")
plt.title('Rent Price Ranges by Property Type')
plt.xlabel('Price (Naira)')
plt.ylabel('Property Type')
plt.ticklabel_format(style='plain', axis='x')
plt.tight_layout()
plt.savefig(f'{output_dir}/9_price_by_property_type.png', dpi=300)
plt.close()

# ==========================================
# 3. The Correlation Heatmap (The ultimate Data Science flex)
# ==========================================
print("Generating Correlation Heatmap...")
plt.figure(figsize=(8, 6))

# Select only the numerical columns for the math
numerical_cols = ['price', 'bedrooms', 'bathrooms', 'toilets', 'size_sqm']
correlation_matrix = df[numerical_cols].corr()

# Draw the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Heatmap: What drives the price?')
plt.tight_layout()
plt.savefig(f'{output_dir}/10_correlation_heatmap.png', dpi=300)
plt.close()

print(f"\n✅ Phase 2 Complete! Check your '{output_dir}' folder for graphs 8, 9, and 10.")