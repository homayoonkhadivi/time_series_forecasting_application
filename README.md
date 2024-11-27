# Time Series Smoothing, Forecasting, and Noise Analysis with Streamlit

## Overview
This project demonstrates a **Streamlit** application that integrates **data science**, **machine learning**, and **data engineering** workflows for time series analysis. It includes methods for smoothing, forecasting, and noise analysis, providing interactive visualisations and insights. The app showcases skills in Python programming, MLOps, data engineering, and front-end presentation for data science projects.

## Features
- **Interactive Time Series Smoothing**:
  - LOESS (Locally Estimated Scatterplot Smoothing)
  - Rolling Window Smoothing
  - Exponential Smoothing (EWMA)

- **Time Series Forecasting**:
  - ARIMA model for forecasting future values with configurable periods.

- **Noise Analysis**:
  - Calculates the difference between original and smoothed data.
  - Visualises cumulative noise change over time.

- **Visualisation and Reporting**:
  - Interactive dashboards with **Streamlit**.
  - Plots for original, smoothed, and forecasted time series.
  - Cumulative noise change visualisation.

- **Parameter Customisation**:
  - User-configurable parameters for smoothing methods and forecasting periods.

## Tools and Libraries
- **Streamlit**: For building the interactive application.
- **Pandas**: For data manipulation.
- **Matplotlib**: For creating detailed visualisations.
- **Statsmodels**: For LOESS, ARIMA, and Exponential Smoothing.
- **NumPy**: For numerical computations.
- **Python**: Core programming language for the project.

## Application Workflow
1. **Data Loading**:
   - Loads a CSV file containing time series data.
   - Cleans and preprocesses data for analysis.

2. **Smoothing**:
   - Users can choose from LOESS, Rolling, or Exponential smoothing.
   - Parameters such as window size, alpha, and fraction can be adjusted interactively.

3. **Forecasting**:
   - ARIMA models are applied to both original and smoothed data.
   - Forecasted values are displayed for user-defined periods.

4. **Noise Analysis**:
   - Computes the noise (difference between original and smoothed data).
   - Visualises cumulative noise change over time.

5. **Visualisation**:
   - Plots for the original, smoothed, and forecasted time series.
   - Noise analysis results are displayed in detailed plots.

## Getting Started
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)

### Installation
1. Clone the repository:
  ```
git clone https://github.com/yourusername/streamlit-time-series-analysis.git
cd streamlit-time-series-analysis
```
  
2. Install the required Python libraries:
```
   pip install -r requirements.txt
```
3. Add your time series dataset:
   - Place your CSV file (e.g., ts_frame_sample.csv) in the root directory.

#### Running the Application

```
streamlit run main.py
```
Access the app in your browser at `http://localhost:8501`

### Repository Structure

```
.
├── main.py               # Main Streamlit application file
├── ts_frame_sample.csv   # Sample time series dataset (add your own data)
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```

### Example Use Cases
- Time series analysis for forecasting and trend detection.
- Exploratory data analysis for noisy datasets.
- Visualising and comparing different smoothing techniques.

### Contact
Feel free to reach out if you have questions or suggestions:

Homi Khadivi

[Visit My LinkedIn Profile](https://www.linkedin.com/in/homayoon-khadivi)

[Visit My GitHub](https://github.com/homayoonkhadivi) 
