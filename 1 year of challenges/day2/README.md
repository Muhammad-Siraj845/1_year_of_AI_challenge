# California Housing Price Prediction

A machine learning project that predicts California housing prices using linear regression. This project demonstrates basic machine learning workflows including data loading, model training, evaluation, and visualization.

## Features

- **Data Loading**: Uses the California Housing dataset from scikit-learn
- **Model Training**: Implements Linear Regression for price prediction
- **Model Evaluation**: Calculates RMSE (Root Mean Square Error) and R² score
- **Visualization**: Creates scatter plots comparing actual vs predicted prices
- **Data Splitting**: Implements train-test split for proper model evaluation

## Project Structure

```
day2/
├── app.py                    # Main application with ML pipeline
├── main.py                   # Simple hello world script
├── california_housing_train.csv  # Dataset file
├── predictions_vs_actual.png     # Generated visualization
├── pyproject.toml           # Project configuration
├── uv.lock                  # Dependency lock file
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Using pip

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Using uv (recommended)

1. Clone or download this repository
2. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

### Running the Main Application

To run the machine learning pipeline:

```bash
python app.py
```

This will:
1. Load the California Housing dataset
2. Split the data into training and test sets
3. Train a linear regression model
4. Make predictions on the test set
5. Calculate and display RMSE and R² metrics
6. Generate a visualization saved as `predictions_vs_actual.png`

### Running the Simple Script

To run the basic hello world script:

```bash
python main.py
```

## Output

The application will output:
- **RMSE**: Root Mean Square Error (lower is better)
- **R²**: Coefficient of determination (closer to 1 is better)
- **Visualization**: A scatter plot saved as `predictions_vs_actual.png` showing actual vs predicted house prices

## Dataset

The project uses the California Housing dataset which includes:
- **Features**: 8 different housing-related features (e.g., median income, house age, etc.)
- **Target**: Median house value in California census blocks
- **Size**: Approximately 20,640 samples

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and datasets
- **matplotlib**: Data visualization

## Model Performance

The linear regression model typically achieves:
- RMSE: ~0.7-0.8 (in units of $100,000)
- R²: ~0.6-0.7 (60-70% of variance explained)

## Contributing

Feel free to improve this project by:
- Adding more advanced models (Random Forest, XGBoost, etc.)
- Implementing cross-validation
- Adding feature engineering
- Improving visualizations
- Adding model persistence

## License

This project is open source and available under the MIT License.
