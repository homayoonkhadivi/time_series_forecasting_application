import streamlit as st
import pandas as pd
#from proplib.setup.types import StatType
#from proplib.setup.types import SalesStatisticsIdentifier
#from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#from proplib.areas import AvailableOptimisationProfiles
#from proplib.setup.types import StatType, State
#from proplib.pipelines import GtsOutputData

# Custom pipeline loading function for loading the dataset
def load_pipeline_data():
    df = pd.read_csv("./ts_frame_sample.csv")
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# Define smoothing functions
def loess_smoothing(data: pd.Series, frac: float = 0.05, it: int = 2, delta: float = 0.01) -> pd.Series:
    """Apply LOESS (Locally Estimated Scatterplot Smoothing) to the input time series data."""
    smoothed_data = lowess(data, data.index, frac=frac, it=it, delta=delta, return_sorted=False)
    return pd.Series(smoothed_data, index=data.index)

def rolling_smoothing(data: pd.Series, window: int = 5, center: bool = True, min_periods: int = 1, win_type: str = None) -> pd.Series:
    """Apply a rolling window smoothing method to the input time series data."""
    return data.rolling(window=window, center=center, min_periods=min_periods, win_type=win_type).mean()

def exponential_smoothing_advanced(data: pd.Series, alpha: float = 0.3) -> pd.Series:

    
    return data.ewm(alpha=alpha).mean()
# ARIMA forecasting function
def arima_forecast(data: pd.Series, forecast_period: int = 30):
    """Apply ARIMA forecasting to the input time series data."""
    data = data.dropna()  # Ensure no NaN values before ARIMA
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_period)
    return forecast

# Function to calculate noise change ranges and cumulative sum
def noise_change_analysis(original_series: pd.Series, smoothed_series: pd.Series):
    subtract = original_series - smoothed_series
    subtract = pd.Series(subtract, index=original_series.index)

    start = subtract.index.min() + pd.Timedelta(days=365)
    end = pd.Timestamp("2024-01-01")
    idx = pd.date_range(start, end, freq='D')

    # First Plot: Cumulative noise changes
    cumulative_noise_sum = pd.Series((subtract.loc[:d].sum() for d in pd.date_range(start, end, freq='D')), index=idx)

    return subtract, cumulative_noise_sum

# Main Streamlit app
st.title("Time Series Smoothing and Forecasting with Noise Analysis")

#state = State.SA
#eligible_code_asgs = AvailableOptimisationProfiles[state]["OptimisationTestSa"]["sa1_codes"]
eligible_code_asgs = "254585458"
# Select code and load the corresponding time series
code = st.selectbox("Select a code", eligible_code_asgs)
ts_data = load_pipeline_data()

# Let users select a column to smooth and forecast
columns = ts_data.columns.tolist()
selected_column = st.selectbox("Select a column", columns)

# Show the original time series data
st.write(f"Original Time Series for Code {code} - Column: {selected_column}")
st.line_chart(ts_data[selected_column])

# Smoothing method selection
smoothing_method = st.selectbox("Select Smoothing Method", ["Loess", "Rolling", "Exponential"])

# Parameters for smoothing methods
if smoothing_method == "Loess":
    frac = st.slider(
        "LOESS Fraction (Controls the amount of smoothing)", 
        0.0001, 0.5, 0.05,
        help="The fraction of the data used to fit each local regression. Lower values give more local detail."
    )
    it = st.slider(
        "LOESS Iterations", 
        1, 10, 2, 
        help="The number of iterations to perform. Higher iterations result in more robust smoothing."
    )
    delta = st.slider(
        "LOESS Delta (Robustification distance)", 
        0.0, 1.0, 0.01,
        help="Distance used to robustify the weights for outliers. Higher values make it more tolerant of outliers."
    )
    smoothed_series = loess_smoothing(ts_data[selected_column], frac=frac, it=it, delta=delta)

elif smoothing_method == "Rolling":
    window = st.slider(
        "Rolling Window (Size of the moving window)", 
        3, 180, 5,
        help="The size of the moving window used for smoothing. A larger window gives more smoothing."
    )
    center = st.checkbox(
        "Center the rolling window", 
        value=True, 
        help="If checked, the window is centered on the data point. Otherwise, the window will be right-aligned."
    )
    min_periods = st.slider(
        "Minimum Periods", 
        1, window, 1, 
        help="Minimum number of observations in the window required to have a value."
    )
    win_type = st.selectbox(
        "Window Type", 
        [None, 'boxcar', 'triang', 'blackman', 'hamming', 'bartlett'],
        help="The type of window function to apply. None applies a simple moving average."
    )
    smoothed_series = rolling_smoothing(ts_data[selected_column], window=window, center=center, min_periods=min_periods, win_type=win_type)

else:  # Exponential Smoothing (EWMA)
    alpha = st.slider(
        "Alpha (Smoothing factor for the level)", 
        0.01, 1.0, 0.3, 
        help="Controls the smoothing factor for the level. Higher values discount older observations faster."
    )
    
    # Apply exponential smoothing using pandas' EWM function
    smoothed_series = exponential_smoothing_advanced(ts_data[selected_column], alpha=alpha)
# Plot the smoothed series
st.write(f"Smoothed Time Series for Code {code} - {smoothing_method}")
st.line_chart(smoothed_series)

# ARIMA forecasting for both original and smoothed series
forecast_period = st.slider("Select Forecast Period (Days)", 10, 100, 30)

# Forecast without smoothing
forecast_original = arima_forecast(ts_data[selected_column], forecast_period)

# Forecast with smoothing
forecast_smoothed = arima_forecast(smoothed_series.dropna(), forecast_period)

# Plot original, smoothed data, and forecasts
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ts_data[selected_column], label='Original Data')
ax.plot(smoothed_series, label=f'Smoothed Data ({smoothing_method})')
ax.plot(forecast_original.index, forecast_original, label='Forecast (Original)', linestyle='--')
ax.plot(forecast_smoothed.index, forecast_smoothed, label=f'Forecast ({smoothing_method})', linestyle='--')
ax.set_title(f'Time Series and Forecasting for Code {code} - {selected_column}')
ax.legend()
st.pyplot(fig)

# Noise change range analysis (Subtract and Cumulative Noise)
subtract, cumulative_noise = noise_change_analysis(ts_data[selected_column], smoothed_series)

# Plot original, smoothed, and subtract series
st.write(f"Original, Smoothed, and Subtracted Series for Code {code}")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ts_data[selected_column], label='Original Data')
ax.plot(smoothed_series, label=f'Smoothed Data ({smoothing_method})')
ax.plot(subtract, label='Subtract (Original - Smoothed)')
ax.set_title(f'Original, Smoothed, and Subtracted Series for Code {code}')
ax.legend()
st.pyplot(fig)

# Plot cumulative sum of noise change (second plot)
st.write(f"Cumulative Sum of Noise Change for Code {code}")
fig, ax = plt.subplots()
ax.plot(cumulative_noise.index, cumulative_noise, label='Cumulative Sum of Noise Change')
ax.set_title(f'Cumulative Sum of Noise Change for Code {code}')
ax.set_xlabel('Date')
ax.set_ylabel('Cumulative Sum')
ax.legend()
st.pyplot(fig)
