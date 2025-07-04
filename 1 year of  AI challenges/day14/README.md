# PCA on Iris Dataset

This project demonstrates Principal Component Analysis (PCA) on the classic Iris dataset using Python. The code reduces the dimensionality of the dataset to two principal components and visualizes the results.

## Overview
- **Dataset:** The Iris dataset (`iris.csv`)
- **Goal:** Apply PCA to reduce features to 2D for visualization and analysis
- **Output:**
  - A scatter plot of the first two principal components colored by species (`pca_plot.png`)
  - A CSV file with the PCA components and species labels (`iris_pca.csv`)

## How It Works
1. **Load Data:** Reads `iris.csv` and assigns column names.
2. **Preprocessing:** Drops the species column for unsupervised PCA, but keeps it for visualization.
3. **Scaling:** Standardizes features using `StandardScaler`.
4. **PCA:** Reduces the dataset to two principal components.
5. **Visualization:**
    - Plots the data in the new 2D PCA space, colored by species.
    - Saves the plot as `pca_plot.png`.
6. **Save Results:**
    - Exports the PCA-transformed data with species labels to `iris_pca.csv`.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

## Usage
Run the analysis with:
```bash
python app.py
```

## Files
- `app.py` — Main script for PCA analysis
- `iris.csv` — Input dataset
- `pca_plot.png` — Output PCA scatter plot
- `iris_pca.csv` — Output CSV with PCA components and species

## Notes
- Make sure `iris.csv` is present in the same directory as `app.py`.
- The script prints the explained variance ratio for each principal component.
