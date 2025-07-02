# Spam Classification with Naive Bayes

This project demonstrates how to use a Naive Bayes classifier to detect spam emails using the UCI Spambase dataset.

## Dataset
- **File:** `spambase.csv`
- **Source:** [UCI Machine Learning Repository - Spambase Data Set](https://archive.ics.uci.edu/ml/datasets/spambase)
- **Description:** Each row represents an email, with 57 features (word and character frequencies, etc.) and a `label` column (1 = spam, 0 = not spam).

## Project Structure
- `app.py`: Main script for loading data, training the model, evaluating, and visualizing results.
- `spambase.csv`: The dataset used for training/testing.
- `confusion_matrix.png`: Output image showing the confusion matrix.
- `spambase_copy.csv`: A copy of the dataset saved after running the script.

## Setup
1. **Clone the repository or download the files.**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script:
```bash
python app.py
```

## What the Script Does
1. Loads the dataset from `spambase.csv`.
2. Splits the data into training and test sets.
3. Trains a Gaussian Naive Bayes classifier.
4. Evaluates the model and prints the accuracy.
5. Plots and saves the confusion matrix as `confusion_matrix.png`.
6. Saves a copy of the dataset as `spambase_copy.csv`.

## Output
- **Accuracy**: Printed in the terminal.
- **Confusion Matrix**: Saved as `confusion_matrix.png`.
- **Dataset Copy**: Saved as `spambase_copy.csv`.

## Requirements
See `requirements.txt` for the full list of dependencies.

## Notes
- The script expects `spambase.csv` to have a header row with column names.
- The target column for classification is `label` (1 = spam, 0 = not spam).
- If you encounter errors, ensure your dataset matches the expected format.

## License
This project is for educational purposes.
