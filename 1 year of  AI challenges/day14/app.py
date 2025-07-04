import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
y = df['species']  # For visualization only

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: {explained_variance}")
print(f"Total Explained Variance: {sum(explained_variance):.2f}")

# Visualize PCA components
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', s=100)
plt.xlabel(f"PC1 ({explained_variance[0]:.2%} variance)")
plt.ylabel(f"PC2 ({explained_variance[1]:.2%} variance)")
plt.title("PCA on Iris Dataset")
plt.savefig("pca_plot.png")
plt.close()

# Save dataset with PCA components
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['species'] = y
df_pca.to_csv("iris_pca.csv", index=False)