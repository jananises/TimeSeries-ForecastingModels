# -*- coding: utf-8 -*-
"""
Created on Sat May 16 15:03:23 2020

@author: Janani
"""
# Time Series Forecasting for Airlines - Passengers Dataset
#Load Librraies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


#Read the dataset
passengers=pd.read_excel('C:\\Data Science\\Assignments\\Forecasting\\Airlines+Data.xlsx')
#To make plotting graphs easier, we set the index of dataframe to the Month. During plots, it will be easier
passengers['Month'] = pd.to_datetime(passengers['Month'],infer_datetime_format=True)
indexed_dataset = passengers.set_index(['Month'])
indexed_dataset.head()

print(type(passengers))
print(type(indexed_dataset))
print(passengers.size)

#Clean the data, check for any NUll values
passengers.isna().any()

#Statistical analysis
passengers.describe()    #Observations - check for any outliers, no outliers present

#Histogram Plot
passengers.hist()

#Plot Graph to check the trend, seasonality, etc.
plt.xlabel("Date")
plt.ylabel("Air Passengers")
plt.plot(indexed_dataset)   #Observation - Trend is present and seasonality is also present. This is an upward trent

#Feature Engineering
#Creating a Multi-Variate Series/ Dummy Variables for Statistical Analysis
passengers['lag1'] = passengers['Passengers'].shift(1)
passengers['lag2'] = passengers['Passengers'].shift(2)
passengers['lag3'] = passengers['Passengers'].shift(3)
passengers.head()

passengers['MA3'] = passengers['Passengers'].rolling(window=3).mean()
passengers['MA4'] = passengers['Passengers'].rolling(window=4).mean()
passengers['MA5'] = passengers['Passengers'].rolling(window=5).mean()
passengers['Max5'] = passengers['Passengers'].rolling(window=5).max()
passengers['Min5'] = passengers['Passengers'].rolling(window=5).min()
passengers.head()


passengers['year'] = passengers['Month'].dt.year
passengers['Mon'] = passengers['Month'].dt.month

passengers.head(24)

# Create a Base Model/Naive Model

#Create a baseline Model/Naive Model
passengers_base = passengers[["Month","Passengers"]]
passengers_base.head()


#Create a baseline Model/Naive Model
passengers_base = pd.concat([passengers_base, passengers_base['Passengers'].shift(1)], axis=1)
passengers_base.head()


passengers_base.columns = ['Month','Actual_Passengers', 'Forecast_Passengers']

passengers_base.dropna(inplace=True)
passengers_base.head()

#Plot
passengers_base[['Actual_Passengers','Forecast_Passengers']].plot(figsize=(12,8))

from sklearn.metrics import mean_squared_error
passengers_errors = mean_squared_error(passengers_base.Actual_Passengers, passengers_base.Forecast_Passengers)
print("MSE",passengers_errors)
print("RMSE - Naive Model", np.sqrt(passengers_errors))   #RMSE Value - 23.33

#Test for Stationarity - observing through Plots - Analysis 1
#Stationarity Test to apply ARIMA, AR, MA models
rol_mean = indexed_dataset.rolling(window=12).mean()
rol_std = indexed_dataset.rolling(window=12).std()
print(rol_mean, rol_std)


#Plotting Rolling Statistcs
orig = plt.plot(indexed_dataset, color='blue', label="org")
mean = plt.plot(rol_mean, color="green", label="Rolling Mean")
std = plt.plot(rol_std, color="red", label="Rolling Std")
plt.legend(loc='best')
plt.title("Rolling Mean and Std")


### Test For Stationarity - Dicky Fuller Test - Analysis 2
from statsmodels.tsa.stattools import adfuller
print('Results of Dickey Fuller Test:')

dftest = adfuller(indexed_dataset['Passengers'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)    
if dftest[1] <= 0.05:
   print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
else:
   print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
   
# Data Transformation to achieve Stationarity 
#############################################
#Log Scale Transformation, estimating Trend
indexed_dataset_log = np.log(indexed_dataset)
plt.plot(indexed_dataset_log)

#The below transformation is required to make series stationary
movingAverage = indexed_dataset_log.rolling(window=12).mean()
movingSTD = indexed_dataset_log.rolling(window=12).std()
plt.plot(indexed_dataset_log)
plt.plot(movingAverage, color='red')
#plt.plot(movingSTD, color='black')


datasetLogScaleMinusMovingAverage = indexed_dataset_log - movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#Remove NAN values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)

#Write function
def test_stationarity(timeseries):    
    #Determine rolling statistics
    movingAverage = timeseries.rolling(window=12).mean()
    movingSTD = timeseries.rolling(window=12).std()
    
    #Plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey–Fuller test:
    print('Results of Dickey Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
   
   
test_stationarity(datasetLogScaleMinusMovingAverage) #Observation - p-value has coem down from 0.99 to 0.32, still it is non-stationary   

#Applying Exponential Decay Transformation 
exponentialDecayWeightedAverage = indexed_dataset_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(indexed_dataset_log)
plt.plot(exponentialDecayWeightedAverage, color='red')



datasetLogScaleMinusExponentialMovingAverage = indexed_dataset_log - exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusExponentialMovingAverage)
#Observation - p-value has decreased to 0.05, this is relatevly stationary



#Applying Time Shift Transformation
datasetLogDiffShifting = indexed_dataset_log - indexed_dataset_log.shift()
plt.plot(datasetLogDiffShifting)


datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)
#Observation - p-value of 0.07 is not as good as 0.05 of exponential decay



#Now we break down the 3 components of the log scale series 
#Then we simply ignore trend & seasonality and check on the nature of the residual part.
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(indexed_dataset_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexed_dataset_log, label="original")
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label="Trend")
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label="seasonal")
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label="residual")
plt.legend(loc='best')
plt.tight_layout()



decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
decomposedLogData

test_stationarity(decomposedLogData)


#Plotting ACF & PACF plots to arrive at p and q values
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')

#plot ACF
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Auto correlation function')


#plot PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='gray')
plt.title('Partial Auto correlation function')
plt.tight_layout()


# Build Models
import warnings
warnings.filterwarnings("ignore")
#Build AR Model
#AR Model order = p,d,q =0   gives RSS=0.9508
#making order=(2,1,0) 
from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(indexed_dataset_log, order=(2,1,0))
results_AR = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_AR.fittedvalues - datasetLogDiffShifting['Passengers'])**2))
print('Plotting AR model')
print('RMSE: %.4f'%np.sqrt(mean_squared_error(results_AR.fittedvalues, datasetLogDiffShifting['Passengers'])))


#MA Model # RSS = 0.8278
model = ARIMA(indexed_dataset_log, order=(0,1,2))
results_MA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_MA.fittedvalues - datasetLogDiffShifting['Passengers'])**2))
print('Plotting MA model')
print('RMSE: %.4f'%np.sqrt(mean_squared_error(results_MA.fittedvalues, datasetLogDiffShifting['Passengers'])))


# AR+I+MA = ARIMA model   #0.6931
model = ARIMA(indexed_dataset_log, order=(2,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - datasetLogDiffShifting['Passengers'])**2))
print('Plotting ARIMA model')
print('RMSE: %.4f'%np.sqrt(mean_squared_error(results_ARIMA.fittedvalues, datasetLogDiffShifting['Passengers'])))


results_ARIMA.summary()
results_ARIMA.aic # -185.31

#Hyper Parameters tuning
#Check for the best ARIMA order - hyperparameter tuning
p_values = range(0,5)
d_values = range(0,5)
q_values = range(0,5)

for p in p_values:
  for d in d_values:
    for q in q_values:
      order = (p,d,q)
      train,test = indexed_dataset[0:84], indexed_dataset[84:96]
      predictions = list()
      for i in range(len(test)):
        try:
          model = ARIMA(train,order)
          model_fit = model.fit(disp=0)
          pred_y = model_fit.forecast()[0]
          predictions.append(pred_y)
          error = mean_squared_error(test, predictions)
          print('MSE', order,error)
        except:
          continue
 #Order - order=(4,1,2) is the best
      
# AR+I+MA = ARIMA model  Best Model-  #0.6792, MSE - 2322.099
model = ARIMA(indexed_dataset_log, order=(4,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - datasetLogDiffShifting['Passengers'])**2))
print('Plotting ARIMA model')
print('RMSE: %.4f'%np.sqrt(mean_squared_error(results_ARIMA.fittedvalues, datasetLogDiffShifting['Passengers'])))        
        
results_ARIMA.aic # -183.17754058849272 (AIC value has reduced)

#Prediction & Reverse transformations
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())


#Convert to cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum)


predictions_ARIMA_log = pd.Series(indexed_dataset_log['Passengers'].iloc[0], index=indexed_dataset_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()


# Inverse of log is exp.
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(indexed_dataset)
plt.plot(predictions_ARIMA)


indexed_dataset_log

#We have 96(existing data of 8 yrs in months) data points. 
#And we want to forecast for additional 96 data points or 8 yrs.
results_ARIMA.plot_predict(1,192) 
#results_ARIMA.forecast(steps=120)

#Final
x = results_ARIMA.forecast(steps=12)
print(x)        

