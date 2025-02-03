import pandas as pd
from pycaret.clustering import setup, create_model, assign_model

# Load dataset
df = pd.read_excel("GHED_data.XLSX", sheet_name="Data")

# Select relevant columns
selected_columns = ['location', 'region', 'income', 'year', 'che_gdp', 'che_pc_usd', 'che', 'gghed', 'pvtd', 'gdp_usd2022_pc']
df = df[selected_columns]

# Generate missing value report BEFORE filling
missing_before = df.isnull().sum()
print("🔍 Missing Values BEFORE Filling:")
print(missing_before[missing_before > 0])

# Fill missing values
df.ffill(inplace=True)  # Forward Fill for small gaps
df['gghed'].fillna(df['gghed'].median(), inplace=True)  # Median fill for larger gaps
df['pvtd'].fillna(df['pvtd'].median(), inplace=True)  # Median fill for larger gaps
df['gdp_usd2022_pc'].fillna(df['gdp_usd2022_pc'].median(), inplace=True)  # Median fill for GDP per capita

# Generate missing value report AFTER filling
missing_after = df.isnull().sum()
print("\n✅ Missing Values AFTER Filling:")
print(missing_after[missing_after > 0])  # Should be empty if all missing values are filled

# Initialize PyCaret Clustering
clustering_setup = setup(data=df[['che_gdp', 'che_pc_usd', 'che', 'gghed', 'pvtd', 'gdp_usd2022_pc']], normalize=True, session_id=42)

# Apply K-Means Clustering (Choosing k=4 clusters)
kmeans = create_model('kmeans', num_clusters=4)

# Assign cluster labels
df_clusters = assign_model(kmeans)

# Merge cluster labels with original dataset
df['Cluster'] = df_clusters['Cluster']

# Save the processed data with clusters
df.to_csv("health_expenditure_clusters.csv", index=False)

print("\n✅ Data Processed & Missing Values Filled")
