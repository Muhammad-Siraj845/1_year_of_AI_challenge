# IMDB Sentiment Analysis

This project implements a machine learning model to classify IMDB movie reviews as positive or negative using Natural Language Processing (NLP) techniques.

## Features

- **Text Preprocessing**: Removes HTML tags, converts to lowercase, tokenizes, and removes stop words
- **TF-IDF Vectorization**: Converts text to numerical features using TF-IDF
- **Logistic Regression**: Trains a binary classification model
- **Model Evaluation**: Provides accuracy metrics and confusion matrix visualization
- **Data Visualization**: Generates a confusion matrix heatmap

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for package dependencies

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have the `IMDB-Dataset.csv` file in the project directory
2. Run the application:

```bash
python app.py
```

The script will:
- Load and preprocess the IMDB dataset
- Train a logistic regression model
- Evaluate the model performance
- Generate a confusion matrix visualization (`confusion_matrix.png`)
- Save the processed dataset

## Dataset

The project expects an `IMDB-Dataset.csv` file with the following columns:
- `review`: Text content of the movie review
- `sentiment`: Target variable ('positive' or 'negative')

## Output

- **Console Output**: Model accuracy score
- **confusion_matrix.png**: Visualization of the confusion matrix
- **IMDB-Dataset.csv**: Updated dataset with preprocessed reviews

## Model Details

- **Algorithm**: Logistic Regression
- **Features**: TF-IDF vectorization (max 5000 features)
- **Text Preprocessing**: HTML tag removal, lowercase conversion, tokenization, stop word removal
- **Train/Test Split**: 80/20 split with random state 42

## Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `scikit-learn`: Machine learning algorithms and utilities
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical data visualization
- `nltk`: Natural Language Processing toolkit

## License

This project is part of the "1 year of AI challenges" series.
