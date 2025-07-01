# Telco Customer Churn Prediction

This project predicts customer churn using a K-Nearest Neighbors (KNN) classifier. It preprocesses the data, trains a model, evaluates its accuracy, and visualizes the results with a confusion matrix.

## Dataset
- The code expects a file named `churn.csv` in the same directory. This should contain the customer data, including a `Churn` column as the target variable.

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python app.py
```

- The script will preprocess the data, train a KNN model, evaluate its accuracy, and save a confusion matrix plot as `confusion_matrix.png`.
- The processed dataset will be saved back to `churn.csv`.

## Output
- `confusion_matrix.png`: Visualization of the confusion matrix for model predictions.
- `churn.csv`: The processed dataset (categorical columns encoded).
