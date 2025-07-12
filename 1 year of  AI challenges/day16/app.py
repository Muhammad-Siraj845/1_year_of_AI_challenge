import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Load dataset
try:
    df = pd.read_csv("Stock_data.csv", parse_dates=['Date'], index_col='Date')
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Use closing price
series = df['Close']

# Split data
train_size = int(len(series) * 0.8)
train, test = series[:train_size], series[train_size:]

# Fit ARIMA model (order=(5,1,0) for simplicity)
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"RMSE: {rmse:.2f}")

# Plot forecast vs actual
plt.figure(figsize=(10, 6))
plt.plot(train.index, train, label='Train', color='#1f77b4')
plt.plot(test.index, test, label='Test', color='#ff7f0e')
plt.plot(test.index, forecast, label='Forecast', linestyle='--', color='#2ca02c')
plt.xlabel("Date")
plt.ylabel("Stock Price (USD)")
plt.title("ARIMA Forecast vs Actual")
plt.legend()
plt.savefig("forecast.png")
plt.close()

# Save forecast
forecast_df = pd.DataFrame({'Forecast': forecast}, index=test.index)
forecast_df.to_csv("forecast.csv")