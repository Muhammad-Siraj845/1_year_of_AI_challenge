# Stock Price Forecasting with ARIMA

A Python-based stock price forecasting application that uses ARIMA (Autoregressive Integrated Moving Average) models to predict stock prices. This project demonstrates time series analysis and forecasting techniques applied to financial data.

## 📊 Features

- **Time Series Analysis**: Loads and processes historical stock data
- **ARIMA Modeling**: Implements ARIMA(5,1,0) model for price forecasting
- **Data Visualization**: Creates comprehensive plots comparing actual vs predicted values
- **Performance Metrics**: Calculates RMSE (Root Mean Square Error) for model evaluation
- **Export Capabilities**: Saves forecasts to CSV and generates visualization plots

## 📁 Project Structure

```
day16/
├── app.py              # Main application script
├── Stock_data.csv      # Historical stock price data
├── forecast.csv        # Generated forecast results
├── forecast.png        # Visualization plot
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project configuration
├── uv.lock            # Dependency lock file
└── README.md          # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.11 or higher
- pip or uv package manager

### Setup Instructions

1. **Clone or download the project files**

2. **Install dependencies using pip:**
   ```bash
   pip install -r requirements.txt
   ```

   **Or using uv (recommended):**
   ```bash
   uv sync
   ```

3. **Verify installation:**
   ```bash
   python -c "import pandas, numpy, statsmodels, matplotlib, seaborn, sklearn; print('All dependencies installed successfully!')"
   ```

## 📈 Data Format

The application expects a CSV file (`Stock_data.csv`) with the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| Date | Trading date (YYYY-MM-DD) | 2018-09-28 |
| Open | Opening price | 234.05 |
| High | Highest price of the day | 235.95 |
| Low | Lowest price of the day | 230.2 |
| Last | Last traded price | 233.5 |
| Close | Closing price | 233.75 |
| Total Trade Quantity | Number of shares traded | 3069914 |
| Turnover (Lacs) | Trading volume in lakhs | 7162.35 |

## 🚀 Usage

### Running the Application

```bash
python app.py
```

### What the Application Does

1. **Data Loading**: Reads the `Stock_data.csv` file and parses dates
2. **Data Preparation**: Uses closing prices for analysis
3. **Train-Test Split**: Splits data into 80% training and 20% testing sets
4. **Model Training**: Fits an ARIMA(5,1,0) model to the training data
5. **Forecasting**: Generates predictions for the test period
6. **Evaluation**: Calculates RMSE between actual and predicted values
7. **Visualization**: Creates a plot showing train, test, and forecast data
8. **Output**: Saves results to `forecast.csv` and `forecast.png`

### Output Files

- **forecast.csv**: Contains the predicted stock prices for the test period
- **forecast.png**: Visualization showing actual vs predicted stock prices

## 📊 Model Details

### ARIMA Model Configuration

- **Order**: (5, 1, 0)
  - **p=5**: Autoregressive terms (uses 5 previous values)
  - **d=1**: Differencing order (makes series stationary)
  - **q=0**: Moving average terms (not used in this model)

### Model Selection Rationale

- **AR(5)**: Captures short-term price momentum and patterns
- **I(1)**: First differencing removes trend and makes series stationary
- **MA(0)**: No moving average component for simplicity

## 📈 Performance Metrics

The application calculates and displays:

- **RMSE (Root Mean Square Error)**: Measures prediction accuracy
  - Lower values indicate better predictions
  - Formula: √(Σ(actual - predicted)² / n)

## 🔧 Customization

### Modifying Model Parameters

To change the ARIMA model order, edit line 25 in `app.py`:

```python
# Current: ARIMA(5,1,0)
model = ARIMA(train, order=(5,1,0))

# Example: Try ARIMA(3,1,1)
model = ARIMA(train, order=(3,1,1))
```

### Adjusting Train-Test Split

To change the split ratio, modify line 20:

```python
# Current: 80% train, 20% test
train_size = int(len(series) * 0.8)

# Example: 70% train, 30% test
train_size = int(len(series) * 0.7)
```

### Using Different Price Data

To use different price columns, modify line 17:

```python
# Current: Using closing prices
series = df['Close']

# Example: Using opening prices
series = df['Open']
```

## 🐛 Troubleshooting

### Common Issues

1. **File Not Found Error**
   - Ensure `Stock_data.csv` is in the same directory as `app.py`
   - Check file permissions

2. **Import Errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (requires 3.11+)

3. **Memory Issues**
   - For large datasets, consider reducing the data size
   - Close other applications to free up memory

4. **Plot Not Saving**
   - Ensure write permissions in the directory
   - Check if `forecast.png` is open in another application

### Error Messages

- **"Error loading dataset"**: Check CSV file format and location
- **"RMSE: [value]"**: Normal output showing model performance
- **Warnings**: Usually safe to ignore (warnings are suppressed)

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0.0 | Data manipulation and analysis |
| numpy | ≥1.24.0 | Numerical computations |
| statsmodels | ≥0.14.0 | Time series modeling (ARIMA) |
| matplotlib | ≥3.7.0 | Data visualization |
| seaborn | ≥0.12.0 | Enhanced plotting |
| scikit-learn | ≥1.3.0 | Machine learning utilities |

## 🤝 Contributing

To improve this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🔮 Future Enhancements

Potential improvements for this project:

- **Multiple Models**: Compare ARIMA with other models (Prophet, LSTM)
- **Hyperparameter Tuning**: Automatically find optimal ARIMA parameters
- **Real-time Data**: Integrate with live stock APIs
- **Web Interface**: Create a web dashboard for visualization
- **Portfolio Analysis**: Extend to multiple stocks
- **Risk Assessment**: Add confidence intervals and risk metrics

## 📞 Support

For questions or issues:

1. Check the troubleshooting section above
2. Review the code comments for clarification
3. Open an issue in the repository

---

**Note**: This is a demonstration project for educational purposes. Stock price predictions should not be used as the sole basis for investment decisions. Always conduct thorough research and consider consulting with financial advisors.
