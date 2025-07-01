# Telco Customer Churn Prediction

This project predicts customer churn using a voting ensemble of Logistic Regression, Decision Tree, and K-Nearest Neighbors classifiers. The dataset is preprocessed, models are trained, and results are visualized with a confusion matrix.

## Dataset
- The dataset should be named `churn.csv` and placed in the project directory.

## Requirements
See `requirements.txt` for the list of dependencies.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   python app.py
   ```
3. The script will output the accuracy and save a confusion matrix plot as `confusion_matrix.png`.

## Output
- `confusion_matrix.png`: Visualization of the confusion matrix.
- The processed dataset is saved back to `churn.csv`.
