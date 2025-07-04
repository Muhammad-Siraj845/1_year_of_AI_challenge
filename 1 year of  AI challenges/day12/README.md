# Iris K-Means Clustering

This project performs K-Means clustering on the classic Iris dataset using Python. It preprocesses the data, applies clustering, evaluates the result with a silhouette score, visualizes the clusters, and saves the clustered data to a new CSV file.

## Features
- Loads the Iris dataset from a CSV file
- Scales features for better clustering
- Applies K-Means clustering (k=3)
- Evaluates clustering with silhouette score
- Visualizes clusters and centroids
- Saves the clustered data to `iris_with_clusters.csv`

## Requirements
- Python 3.7+
- See `requirements.txt` for Python package dependencies

## Installation
1. Clone this repository or download the code files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure `iris.csv` is present in the same directory as `app.py`.
2. Run the script:
   ```bash
   python app.py
   ```
3. The script will output the silhouette score, save a cluster plot as `clusters.png`, and save the clustered data as `iris_with_clusters.csv`.

## Notes
- If your `iris.csv` does not have a header row, the script will assign the correct column names automatically.
- The script uses only the four numeric features for clustering.
