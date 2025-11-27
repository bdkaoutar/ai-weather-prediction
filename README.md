AI Weather Prediction – Temperature Forecasting Using Machine Learning
Predicting monthly average temperature using 11 years of weather data (2012–2024)

This project applies data science and machine learning techniques to predict the average daily temperature using meteorological datasets collected from Seoul between 2012 and 2024.
It includes full data preprocessing, feature engineering, model training, evaluation, and visual analysis.

Project Overview

With the increasing impact of climate variability, accurate weather prediction has become critical for several sectors including agriculture, energy management, and urban planning.
This project explores multiple regression algorithms to forecast temperature and compare their performance.

The workflow includes:

✓ Collecting and cleaning raw meteorological data

✓ Handling missing values and formatting timestamps

✓ Feature engineering (lags, moving averages, time-based features)

✓ Training machine learning models

✓ Evaluating models with RMSE, MAE, and R²

✓ Visualizing long-term temperature, precipitation, and snowfall trends

Dataset

The dataset consists of 11 years (2012–2024) of weather data including:

Temperature (min / max / avg)

Humidity

Precipitation

Snowfall

Solar radiation & energy

Wind speed / gust

Weather conditions

Files are stored in CSV format and merged into a single dataset during preprocessing.

Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib

Glob

Jupyter Notebook (optional)

Modeling

Two regression models were trained:

Linear Regression

R²: 0.94

RMSE: 2.62

MAE: 1.72

Best performance overall

Strong indication that temperature trends are mostly linear in this dataset

Random Forest Regressor

R²: 0.66

RMSE: 6.19

MAE: 3.74

Lower performance, suggesting limited nonlinear relationship in data

Linear Regression proved to be the most accurate model for this dataset.

Visualizations

The project includes multiple plots:

Temperature min/max cycles (2012–2024)

Monthly/annual precipitation

Snowfall frequency

Anomaly spikes (e.g., extreme rainfall in 2022)

Predicted vs. actual temperature comparisons

Error distribution histograms

Correlation heatmap

Preprocessed dataset preview

These help analyze long-term climate trends and verify model behavior.

How It Works

Run the main script:

python main.py


Steps performed:

Load and merge all CSV files

Clean the data and fix missing values

Convert timestamps and optimize dtypes

Feature engineering

Model training and evaluation

Generate all plots

Export cleaned dataset

Project Structure
ai-weather-prediction/
│
├── main.py
├── weather.csv
├── *.csv                        # Raw weather files (2012–2024)
├── error_distr.png
├── temp_predic_real.png
├── temp_min_max.png
├── snow.png
├── precip.png
├── tot_precip.png
├── AI Project Weather.pdf       # Full project report
└── weather AI project.pptx

Results Summary

Temperature predictions follow real values with strong accuracy

Seasonal patterns clearly visible

Linear regression best fits historical weather trends

Notable anomaly: extreme rainfall in 2022

Feature engineering significantly improved model accuracy

Future Improvements

Add more non-linear models (XGBoost, LSTM)

Extend dataset beyond 2024

Build a web dashboard for interactive forecasting

Deploy model via API or cloud service

Authors

Kaoutar Boudribil

Samya Habib

Hala Rami
