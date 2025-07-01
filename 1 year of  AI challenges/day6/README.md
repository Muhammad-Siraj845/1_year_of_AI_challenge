# Telco Customer Churn Prediction

This project predicts customer churn using a Random Forest Classifier on the Telco Customer Churn dataset. It includes preprocessing, handling class imbalance with SMOTE, model training, evaluation, and visualization of results.

## Features
- Preprocessing of categorical variables
- Handling class imbalance using SMOTE
- Model training with Random Forest
- Evaluation using accuracy and confusion matrix
- Visualization of confusion matrix

## Dataset
The dataset should be named `churn.csv` and placed in the same directory as `app.py`. (You can use the provided `Telco-Customer-Churn.csv` by renaming it to `churn.csv`.)

## Setup
1. Clone the repository or download the files.
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code
1. Ensure `churn.csv` is present in the directory.
2. Run the script:
   ```bash
   python app.py
   ```
3. The script will output the model accuracy and save a confusion matrix plot as `confusion_matrix.png`.

## Output
- `confusion_matrix.png`: Visualization of the confusion matrix.
- The script prints the accuracy score in the terminal.

## Notes
- The script will overwrite `churn.csv` with the encoded version after running.
- Make sure to install all dependencies listed in `requirements.txt` before running the script.
