import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1. Load the dataset
# (Update path if necessary, e.g., 'ps1/food_delivery_time.csv')
df = pd.read_csv('C:/Users/akshat/Projects/AIML Workshop/el plan/ps1/food_delivery_time.csv')

# 2. Select features to group by 
# (We'll use distance, prep time, traffic delay, and delivery time)
features = [
    'distance_km', 
    'prep_time_min', 
    'traffic_delay_min', 
    'delivery_time_min'
]

X = df[features]

# 3. Handle missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 4. Standardize features (crucial for KMeans so features on larger scales don't dominate)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 5. Initialize and fit KMeans (let's find 3 distinct delivery groups/clusters)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 6. Print summary statistics for each cluster
print("--- Cluster Summary Averages ---")
cluster_summary = df.groupby('cluster')[features].mean()
print(cluster_summary)
print("-" * 32)

plt.figure(figsize=(10, 6))

# Scatter plot: Distance vs Total Delivery Time colored by cluster
scatter = plt.scatter(
    df['distance_km'], 
    df['delivery_time_min'], 
    c=df['cluster'], 
    cmap='viridis', 
    alpha=0.7, 
    edgecolors='k'
)

# Plot cluster centers (inverse transformed back to original scale for readability)
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(
    centers_original[:, 0], # distance_km column index
    centers_original[:, 3], # delivery_time_min column index
    c='red', 
    marker='X', 
    s=200, 
    label='Cluster Centers'
)

plt.title('KMeans Clustering of Delivery Orders')
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Time (min)')
plt.legend(handles=scatter.legend_elements()[0], labels=['Cluster 0', 'Cluster 1', 'Cluster 2'])
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()