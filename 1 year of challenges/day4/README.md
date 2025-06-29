# Spam Classification with Logistic Regression

This project uses the Spambase dataset to build a spam classifier using logistic regression. The model predicts whether an email is spam or not based on various word and character frequency features.

## Dataset
- **spambase.csv**: The main dataset used for training and testing.
- **spambase.DOCUMENTATION.csv** and **spambase.names.csv**: Additional documentation and feature names for the dataset.

## How it Works
1. Loads the dataset and assigns column names.
2. Splits the data into training and testing sets.
3. Trains a logistic regression model.
4. Evaluates the model and prints the accuracy.
5. Plots and saves a confusion matrix as `confusion_matrix.png`.

## Usage
1. Make sure all required dependencies are installed (see below).
2. Run the script:
   ```bash
   python app.py
   ```
3. The script will output the accuracy and save a confusion matrix plot.

## Requirements
See `requirements.txt` for the list of dependencies.

## Output
- `confusion_matrix.png`: Visualization of the confusion matrix after model evaluation.

## License
This project is for educational purposes.
