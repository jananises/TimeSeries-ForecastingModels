
"""
Created on Tue May 19 00:42:45 2020

@author: Janani
"""
#Forecasting Coca-Cola Sales Data-set

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#Read the dataset
coke=pd.read_excel('/content/drive/My Drive/Colab Notebooks/Deep Learning/Data Sets/CocaCola_Sales_Rawdata.xlsx')
coke=pd.read_excel('C:\\Data Science\\Assignments\\Forecasting\\CocaCola_Sales_Rawdata.xlsx')
coke.head()

##################Convert - Q1-Q4-YY format to Convert to-Date time format############################
coke['New'] = pd.to_datetime(coke.Quarter.str.replace(r'(Q\d)_(\d+)', r'\2-\1'))
(dt.where(dt <= pd.to_datetime('today'), dt - pd.DateOffset(years=100))
   .dt.strftime('%b-%Y'))

coke.New = coke.New - pd.DateOffset(days=36525)

#print(coke.size)
coke.head()

coke.columns = ['Dummy','Sales','Quarter']


#Clean the data, check for any NUll values
coke.isna().any()

#To make plotting graphs easier, we set the index of dataframe to the Month. During plots, it will be easier
#coke['Quarter'] = pd.to_datetime(coke['Quarter'],infer_datetime_format=True)
indexed_dataset = pd.Series(coke['Sales'].values, index=coke['Quarter'])
indexed_dataset.head()

#Statistical analysis
coke.describe()    #Observations - check for any outliers, no outliers present

#Histogram Plot
coke.hist()

type(indexed_dataset)

#Plot Graph to check the trend, seasonality, etc.
plt.xlabel("Quarter")
plt.ylabel("Coke Sales")
plt.plot(indexed_dataset)   #Observation - Trend is present and seasonality is also present. This is an upward trent

#Create Dummy Variable of t - values 1 to 42
coke["t"] = np.arange(1,43)
coke.tail()

# Create a Base Model/Naive Model
#Create a baseline Model/Naive Model
coke_base = coke[["Quarter","Sales"]]
coke_base.head()

#Create a baseline Model/Naive Model
coke_base = pd.concat([coke_base, coke_base['Sales'].shift(1)], axis=1)
coke_base.head()

coke_base.columns = ['Quarter','Actual_Sales', 'Forecast_Sales']
coke_base.dropna(inplace=True)
coke_base.head()

#Plot of Naive Model
plt.plot(coke_base[['Actual_Sales','Forecast_Sales']])
plt.legend(loc='best')
plt.title("Naive Model")

from sklearn.metrics import mean_squared_error
coke_error_Lag1 = mean_squared_error(coke_base['Actual_Sales'], coke_base['Forecast_Sales'])
print('Naive Model Lag1- MSE %2d, RMSE %2d', coke_error_Lag1, np.sqrt(coke_error_Lag1))   #MSE = 192405.72, RMSE Value - 438.64

#Test for Stationarity - observing through Plots - Analysis 1
rol_mean = indexed_dataset.rolling(window=4).mean()
rol_std = indexed_dataset.rolling(window=4).std()
#print(rol_mean, rol_std)

#Plotting Rolling Statistcs
orig = plt.plot(indexed_dataset, color='blue', label="org")
mean = plt.plot(rol_mean, color="green", label="Rolling Mean")
std = plt.plot(rol_std, color="red", label="Rolling Std")
plt.legend(loc='best')
plt.title("Rolling Mean and Std")

#Observation
#std is constant, mean has a trend, so not stationary

# Time series decomposition plot 
from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts_add = seasonal_decompose(indexed_dataset,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(indexed_dataset,model="multiplicative")
decompose_ts_mul.plot()

import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models

#Train Test Split
train = coke[0:36]
test = coke[36:42]
#print(train.shape)
#print(test.shape)
print(test.index)
print(train.index)

print(test.Sales.shape)

# Creating a functions to calculate RMSE and MAPE values 

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    pred = pred.astype(float)
    org = org.astype(float)
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

# Creating a function to calculate the RMSE value for test data 
def RMSE(pred,org):
    pred = pred.astype(float)
    org = org.astype(float)
    error = mean_squared_error(pred,org)
    temp = np.sqrt(error)
    return temp

####################### L I N E A R MODEL##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_linear))**2))
#MAPE = MAPE(pred_linear,test.Sales) # 18.96
print('Linear Model - RMSE %2d, MAPE %2d', rmse_linear, MAPE(pred_linear,test.Sales))
#RMSE - 667.42, MAPE = 10.44

#########################Simple Exponential Method###########################################
# Simple Exponential Method
ses_model = SimpleExpSmoothing(train["Sales"]).fit()
pred_ses = ses_model.predict(start = test.index[0],end = test.index[-1])
print('Simple Exp Model - RMSE %2d, MAPE %2d', RMSE(pred_ses,test.Sales), MAPE(pred_ses,test.Sales))
#RMSE - 686.68, MAPE - 11.63

#########################Holt Method###########################################
# Holt method 
hw_model = Holt(train["Sales"]).fit()
pred_hw = hw_model.predict(start = test.index[0],end = test.index[-1])
print('Holt Model - RMSE %2d, MAPE %2d', RMSE(pred_hw,test.Sales), MAPE(pred_hw,test.Sales))
#RMSE - 497.52, MAPE - 9.89
##################################################################################

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(train["Sales"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0],end = test.index[-1])
print('Holts winter exponential smoothing - RMSE %2d, MAPE %2d', RMSE(pred_hwe_add_add,test.Sales), MAPE(pred_hwe_add_add,test.Sales))
#RMSE - 230.99, MAPE - 4.784

########################################################################3
# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0],end = test.index[-1])
print('Holts winter ES - Mult n Add - RMSE %2d, MAPE %2d', RMSE(pred_hwe_mul_add,test.Sales), MAPE(pred_hwe_mul_add,test.Sales))
# MAPE - 3.99, RMSE - 227.7

##########################Plot - Visualization ##################################
#Visualization of Forecasted values for Test data set using different methods 
plt.figure(figsize=(10,8))
plt.plot(train.index, train["Sales"], label='Train',color="black")
plt.plot(test.index, test["Sales"], label='Test',color="blue")
plt.plot(pred_linear.index, pred_linear, label='Linear',color="violet")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="brown")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponentialAdd",color="orange")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponentialMul",color="red")
plt.legend(loc='best')
#Observation
##Holts winter exponential smoothing with multiplicative seasonality and additive trend gives a good prediction 
#with additive seasonality and additive trend give average prediction

###########################Plot - Visualization ##################################
#replot 2 of them for more clarity
#Visualization of Forecasted values for Test data set using Holts ExpAdd and Exp Add+Mul 
plt.figure(figsize=(10,6))
plt.plot(train.index, train["Sales"], label='Train',color="black")
plt.plot(test.index, test["Sales"], label='Test',color="blue")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponentialAdd",color="orange")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponentialMul",color="red")
plt.legend(loc='best')

#####################################################################
### Testing For Stationarity - ADF (Augmented Dickey Fuller) Test
#####################################################################
from statsmodels.tsa.stattools import adfuller

#Ho: It is non stationary
#H1: It is stationary
#Create this function
def adfuller_test(timeseries):
    result=adfuller(timeseries)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

#Test for our Series
adfuller_test(indexed_dataset)

#Differencing Approach - Lag1 - Plot
coke['Sales First Difference'] = coke['Sales'] - coke['Sales'].shift(1)
coke['Sales First Difference'].dropna().plot()

#Test Stationarity
adfuller_test(coke['Sales First Difference'].dropna())

#Differencing Approach - Lag12 - Plot
coke['Seasonal First Difference']=coke['Sales']-coke['Sales'].shift(12)
coke['Seasonal First Difference'].dropna().plot()
#coke.head()

adfuller_test(coke['Seasonal First Difference'].dropna())

##########################################################
#Auto ARIMA model works on the efficient Grid Search and Random Search concepts to find the most optimal parameters to find #the best fitting time series model.
#############################################################################
#pip install -q pyramid_arima  - Anaconda command prompt
 
import pyramid
from pyramid.arima import auto_arima

# Lets us use auto_arima from p
auto_arima_model = auto_arima(indexed_dataset,start_p=0,
                              start_q=0,max_p=10,max_q=10,
                              m=4,start_P=0,seasonal=True,
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=True)

auto_arima_model.summary()
#The best performing model from the optimized grid search is the following-
#ARIMA: order=(2,1,2) seasonal_order=(0, 1, 1, 4); AIC=487.539, BIC=498.815

# Create predictions, evaluate on test
period = coke.shape[0] + 6
preds, conf_int = auto_arima_model.predict(n_periods=period, return_conf_int=True)
print(preds)
print(conf_int)
#print(period)

#Plot to check prediction and forecast of the model
plt.figure(figsize=(10,8))
plt.plot(preds, color="orange",label="Forecast")
plt.plot(coke['Sales'][0:42], color="blue", label="Actual")
plt.legend(loc="best")

preds1, conf_int1 = auto_arima_model.predict(n_periods=test.shape[0], return_conf_int=True)
RMSEval = np.sqrt(mean_squared_error(test['Sales'], preds1))
MAPEtemp = np.abs((preds1-test['Sales']))*100/test['Sales']    
MAPEtemp1 = MAPEtemp.mean()   #RMSE 1154.01, MAPE - 21.25
print('RMSE %2d, MAPE %2d', RMSEval, MAPEtemp1)
#Observation Prediction- The magnitude of prediction is on higher scale, seasonality factor present, however the values are different
#We need to go for other models

#######################SARIMA Model#######################
#SARIMA Model
import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(indexed_dataset,
                                order=(2, 1, 2),
                                seasonal_order=(0, 1, 1, 4),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
#############################################

#Observation = p-value < 0.05, this is a good model
# For getting predictions for future we use predict() function 
SARIMApredict = results.predict(start=0,end=41,dynamic=False)
#SARIMApredict.tail()
# Forecast values
SARIMAforecast = results.forecast(steps=6)
#SARIMAforecast
#Combine together - Prediction + forecast
SARIMAdf = pd.concat([SARIMApredict, SARIMAforecast])
#SARIMAdf.tail(6)
Finaldf = pd.DataFrame(data=SARIMAdf, columns=["Sales"])
Finaldf.tail()

####Printing predicted mean and Confidence Intervals###########
SARIMAforecast1 = results.get_forecast(6)
print('Forecast:')
print(SARIMAforecast1.predicted_mean)
print('Confidence intervals:')
print(SARIMAforecast1.conf_int())

#Plot to check prediction and forecast of the model
plt.figure(figsize=(15,10))
plt.plot(SARIMAdf, color="orange",label="Forecast")
plt.plot(indexed_dataset, color="blue", label="Actual")
plt.legend(loc="best")

#Observation Model is a good fit
#Print RMSE and MAPE values
predict_Sarima = Finaldf['Sales'][36:42]
#print(predict_Sarima)
#preds1, conf_int1 = auto_arima_model.predict(n_periods=test.shape[0], return_conf_int=True)
RMSEval = np.sqrt(mean_squared_error(test['Sales'], predict_Sarima))
MAPEtemp = np.abs((preds1-test['Sales']))*100/test['Sales']    
MAPEtemp1 = MAPEtemp.mean()   #RMSE 75.90, MAPE  21.25
print('RMSE %d, MAPE %d', RMSEval, MAPEtemp1)

#Analysis of residuals###################################
#Plotting the Diagnostics Plot of Residuals
results.plot_diagnostics(figsize=(16, 8))
plt.show()

#Observations: Though the residual plot is not that perfect, but model diagnostics suggests that the model residuals are somewhat close to normally distributed.
#The model is a godd fit.