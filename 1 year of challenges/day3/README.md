# California Housing Price Prediction

This project demonstrates a machine learning approach to predict housing prices in California using the California Housing dataset. It includes data preprocessing, model training, evaluation, and visualization.

## Features

- **Data Loading**: Automatically fetches the California Housing dataset from scikit-learn
- **Data Preprocessing**: Scales features using StandardScaler for better model performance
- **Model Training**: Uses Linear Regression to predict house prices
- **Model Evaluation**: Calculates RMSE and R² score to assess model performance
- **Data Visualization**: 
  - Feature correlation heatmap
  - Actual vs Predicted price scatter plot
- **Data Export**: Saves the processed dataset to CSV format

## Requirements

- Python 3.11 or higher
- See `pyproject.toml` for detailed dependency versions

## Installation

1. Clone or download this repository
2. Install dependencies using uv (recommended):
   ```bash
   uv sync
   ```
   
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main application:
```bash
python app.py
```

## Output

The script will generate:
- `california_housing_train.csv` - The processed dataset
- `feature_correlation.png` - Heatmap showing correlations between features
- `predictions_vs_actual_preprocessed.png` - Scatter plot of actual vs predicted prices
- Console output with model performance metrics (RMSE and R²)

## Model Performance

The Linear Regression model typically achieves:
- **RMSE**: ~0.7-0.8 (Root Mean Square Error)
- **R²**: ~0.6-0.7 (Coefficient of determination)

## Dataset

The California Housing dataset contains:
- 8 features (median income, house age, average rooms, etc.)
- Target variable: Median house value (in $100,000s)
- ~20,640 samples

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and utilities
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization

## Project Structure

```
day3/
├── app.py                          # Main application script
├── california_housing_train.csv    # Processed dataset (generated)
├── feature_correlation.png         # Feature correlation heatmap (generated)
├── predictions_vs_actual_preprocessed.png  # Model predictions plot (generated)
├── pyproject.toml                  # Project configuration and dependencies
├── README.md                       # This file
└── uv.lock                         # Lock file for reproducible builds
```

## License

This project is part of a learning series and is open for educational purposes.
