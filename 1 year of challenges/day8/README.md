# Telco Customer Churn Prediction

This project predicts customer churn using a Support Vector Machine (SVM) classifier on the Telco Customer Churn dataset.

## Features
- Preprocesses and encodes categorical variables
- Scales features for SVM
- Trains an SVM model
- Evaluates accuracy and visualizes the confusion matrix

## Requirements
Install dependencies using:
```
pip install -r requirements.txt
```

## Usage
Run the script with:
```
python app.py
```

The script will output the model accuracy and save a confusion matrix plot as `confusion_matrix.png`.

## Dataset
Place your `churn.csv` file in the same directory as `app.py`.
