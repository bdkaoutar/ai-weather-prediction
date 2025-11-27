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
- Precipitation over time  
- Snowfall distribution  
- Annual precipitation spikes (notably 2022)  
- Predicted vs. actual temperature comparison  
- Error distribution plots  
- Correlation heatmap  
- Preview of the cleaned dataset  

---

## How to Run the Project

Run the main script:

```bash
python main.py
