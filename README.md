# 🌦️ AI Weather Prediction – Temperature Forecasting Using Machine Learning

Predicting monthly average temperature using **11 years of weather data (2012–2024)**.

This project applies data science and machine learning techniques to predict the average daily temperature using meteorological datasets collected from Seoul between 2012 and 2024.  
It includes full data preprocessing, feature engineering, model training, evaluation, and visual analysis.

---

##  Project Overview

With the increasing impact of climate variability, accurate weather prediction has become critical for sectors such as agriculture, energy, and urban planning.  
This project explores multiple regression algorithms to forecast temperature and compare their performance.

### **Workflow Summary**
- ✓ Collecting and cleaning raw meteorological data  
- ✓ Handling missing values and formatting timestamps  
- ✓ Feature engineering (lags, moving averages, time-based features)  
- ✓ Training machine learning models  
- ✓ Evaluating models with RMSE, MAE, and R²  
- ✓ Visualizing long-term temperature, precipitation, and snowfall trends  

---

##  Dataset

The dataset consists of **11 years (2012–2024)** of weather data, including:

- Temperature (min / max / avg)  
- Humidity  
- Precipitation  
- Snowfall  
- Solar energy and radiation  
- Wind speed & wind gust  
- Weather conditions  

Files are stored in CSV format and merged into a single DataFrame during preprocessing.

---

## Technologies Used

- **Python**  
- **Pandas, NumPy**  
- **Scikit-learn**  
- **Matplotlib**  
- **Glob**  
- **Jupyter Notebook (optional)**

---

##  Model Training

Two regression models were implemented and compared:

### ** Linear Regression**
- **R²:** 0.94  
- **RMSE:** 2.62  
- **MAE:** 1.72  
- Best performance overall  
- Indicates strong linear behavior in the dataset  

### ** Random Forest Regressor**
- **R²:** 0.66  
- **RMSE:** 6.19  
- **MAE:** 3.74  
- Less effective for this dataset  

**Conclusion:** Linear Regression is the most accurate model for this dataset.

---

## Visualizations

This project includes multiple plots:

- Seasonal temperature cycles (min & max)
  
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/e603b315-2191-4af2-8c3d-0d8053751455" />

- Precipitation over time
  
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/33d8f385-bee2-44f0-bfc6-bef3fa82fa8e" />

- Snowfall distribution

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/0d15a8bd-dd60-47e0-8e3a-9c48278e3cc3" />

- Annual precipitation spikes (notably 2022)
  
<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/4ca497d0-08fc-43b9-aaf9-d8c449516b96" />

- Predicted vs. actual temperature comparison
  
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/235b7be0-d1ab-41ae-a58c-9753fc9a61de" />

- Error distribution plots
  
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/43f3ab7e-d04c-44e6-ab00-b285abbd30ee" />

- Correlation heatmap  
- Preview of the cleaned dataset  

---

## How to Run the Project

Run the main script:

```bash
python main.py
```

This script:

- Loads and merges all CSV files
- Cleans and preprocesses weather data
- Performs feature engineering
- Trains ML models and evaluates them
- Generates all visualizations
- Saves cleaned dataset

## Project Structure

```bash
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
```

## Key Results

* Linear Regression produced highly accurate predictions
* Clear seasonal temperature patterns observed
* Strong correlation among temperature variables
* Significant precipitation anomalies detected (2022 event)
* Feature engineering improved model performance

## Future Improvements

- Add advanced models (XGBoost, Gradient Boosting, LSTMs)
- Integrate more meteorological features
- Deploy as a web-based dashboard
- Add API for real-time predictions

## Authors

- Samya Habib
- Hala Rami
- Kaoutar Boudribila
