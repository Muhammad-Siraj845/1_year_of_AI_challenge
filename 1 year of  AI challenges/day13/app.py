import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
try:
    columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv("iris.csv", header=None, names=columns)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Preprocess: Drop species for unsupervised learning
X = df.drop('species', axis=1)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Evaluate with silhouette score
silhouette = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {silhouette:.2f}")

# Visualize clusters (using first two features for 2D plot)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette='viridis', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.xlabel("Sepal Length (Scaled)")
plt.ylabel("Sepal Width (Scaled)")
plt.title("K-Means Clustering on Iris Dataset")
plt.legend()
plt.savefig("clusters.png")
plt.close()

# Save dataset with cluster labels
df['cluster'] = clusters
df.to_csv("iris.csv", index=False)