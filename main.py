import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import seaborn as sns



csv_files = glob.glob('*.{}'.format('csv'))
print(csv_files)
weather = pd.concat([pd.read_csv(f) for f in csv_files ], ignore_index=True)

print(weather.head())

#set the datetime column as the index of the DataFrame
weather.set_index('datetime', inplace=True)

# print the head and tail of the DataFrame to make sure that all 11 years of data are present.
print(weather.head())
print(weather.tail())

#get statistical summary of the data
print(weather.describe())

#check the data types of the columns
print(weather.info())

# print(weather.loc["2018-01-01", :]
#Drop columns that are not needed
#drop name because we are working on one city
weather.drop(columns=['name', 'feelslikemax', 'feelslikemin', 'severerisk', 'stations'], inplace=True)

#cleaning data
#check missing values
#Adding "/weather.shape[0]" to the end of the code will give you the proportion of missing values in each column.
print(weather.apply(pd.isnull).sum()/weather.shape[0])
#the results show that most of the comulns are not-numll values,
#and therefore, we won't have to drop any columns

# Fill missing values
print(weather["preciptype"].value_counts())
print(pd.isnull(weather["preciptype"]).sum())

#fill all missing values of preciptype with "rain/snow"
weather["preciptype"].fillna("Non ", inplace=True)
#fill all missing values of windgust with 0, considering no windgust was recorded
weather['windgust'].fillna(0, inplace=True)

print(weather.info())

#make sure we have no missing values
print(weather.apply(pd.isnull).sum()/weather.shape[0])

#check if all the datatypes are numerical
print(weather.dtypes)

#convert the columns that are not numerical to numerical
weather['preciptype'] = weather['preciptype'].map({'rain': 1, 'snow': 2, 'rain,snow': 3})
weather['sunrise'] = pd.to_datetime(weather['sunrise'], errors='coerce')
weather['sunset'] = pd.to_datetime(weather['sunset'], errors='coerce')

# Convert sunrise and sunset to datetime objects
weather['sunrise'] = pd.to_datetime(weather['sunrise'], errors='coerce')
weather['sunset'] = pd.to_datetime(weather['sunset'], errors='coerce')

#turning data into categorical data, to make it easier for the model to interpret
# and to optimize the model's performance, and the memory usage of the DataFrame.
categorical_columns = ['conditions', 'description', 'icon']
for col in categorical_columns:
    weather[col] = weather[col].astype('category')

#turn the datetime column into a datetime object, for better manipulation of the data
# we can even use 'weather.index.year' to search for data from a specific year as an example
weather.index = pd.to_datetime(weather.index)




#--------------------------------Move to the Analysis step--------------------------------

#save the cleaned data to a new csv file
# weather.to_csv("weather_cleaned.csv")
weather.to_csv('weather.csv', index=False)

#plot the temperature min and max over time
weather[["tempmin", "tempmax"]].plot(kind='line', figsize=(10, 5), title='Temperature min and max over time')
plt.show()

#plot the precipitation over time
weather["precip"].plot(kind='line', figsize=(10, 5), title='Precipitation over time')
plt.show()

#plot the snow over time
weather["snow"].plot(kind='line', figsize=(10, 5), title='Snow over time')
plt.show()

#How much did it rain each year?
weather.groupby(weather.index.year)["precip"].sum().plot(kind='bar', figsize=(10, 5), title='Total precipitation each year')
plt.show()
# Seoul experienced record-breaking rainfall in 2022, particularly
# in August. On August 8, the city was hit by the heaviest downpour in
# 80 years, with the Dongjak district recording up to 17 inches
# (43 cm) of rain

#create a new dataframe that only has the numerical value for the correlation matrix
numeric_weather = weather.select_dtypes(include=['number'])

# Compute the correlation matrix
correlation_matrix = numeric_weather.corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

# Add a title
plt.title('Correlation Heatmap', fontsize=16)

# Show the plot
plt.show()

# ---------------------------------End of the Analysis step--------------------------------

# ---------------------------------Start of the Modeling step--------------------------------

# Feature extraction
weather['month'] = weather.index.month
weather['day'] = weather.index.day
weather['year'] = weather.index.year

# Lag features
weather['tempmin_lag1'] = weather['tempmin'].shift(1)
weather['tempmax_lag1'] = weather['tempmax'].shift(1)

# Rolling averages
weather['precip_rolling_mean'] = weather['precip'].rolling(window=7).mean()

#Splitting the data into training and testing sets

# Drop rows with NaN values introduced by lag and rolling calculations
weather.dropna(inplace=True)

# Select features and target
features = ['tempmin', 'tempmax', 'humidity', 'precip', 'month', 'day', 'tempmin_lag1', 'tempmax_lag1', 'precip_rolling_mean']
target = 'temp'  # Predicting maximum temperature

X = weather[features]
y = weather[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

#---------------------------------End of the Modeling step--------------------------------
#---------------------------------Start of the Model Training step------------------------

#Linear Regression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

#Random Forest

rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Cross-validation for Linear Regression
linear_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='r2')
print(f"Linear Regression CV R2: {linear_scores.mean():.2f}")

# Cross-validation for Random Forest
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='r2')
print(f"Random Forest CV R2: {rf_scores.mean():.2f}")


#---------------------------------End of the Model Training step------------------------
#---------------------------------Start of the Model Evaluation step------------------------

def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# Evaluate Linear Regression
lr_rmse, lr_mae, lr_r2 = evaluate_model(y_test, lr_predictions)
print(f"Linear Regression - RMSE: {lr_rmse}, MAE: {lr_mae}, R2: {lr_r2}")

# Evaluate Random Forest
rf_rmse, rf_mae, rf_r2 = evaluate_model(y_test, rf_predictions)
print(f"Random Forest - RMSE: {rf_rmse}, MAE: {rf_mae}, R2: {rf_r2}")

#---------------------------------End of the Model Evaluation step------------------------
#---------------------------------Start of the Model Visualisation & Comparison step------------------------
# After training your models and generating predictions:
lr_predictions = lr_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Ensure indices align before plotting

y_test = y_test.reset_index(drop=True)  # Reset the index of y_test
lr_predictions = pd.Series(lr_predictions, index=y_test.index)  # Align predictions with y_test
rf_predictions = pd.Series(rf_predictions, index=y_test.index)

# Plotting the results

plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual', color='blue', linewidth=2)  # Actual data
plt.plot(y_test.index, lr_predictions, label='Linear Regression Predictions', color='orange')  # LR predictions
plt.plot(y_test.index, rf_predictions, label='Random Forest Predictions', color='green')  # RF predictions
plt.legend()
plt.title('Temperature Predictions vs Actual')
plt.show()


#plot the error distribution
errors = y_test - rf_predictions
plt.hist(errors, bins=20, alpha=0.7)
plt.title('Error Distribution (Random Forest)')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

# Tracé de la distribution des erreurs pour la régression linéaire
lr_errors = y_test - lr_predictions
plt.hist(lr_errors, bins=20, alpha=0.7, color='orange')
plt.title('Error Distribution (Linear Regression)')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

