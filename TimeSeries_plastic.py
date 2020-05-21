# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:00:44 2020

@author: Janani
"""
# Time Series Forecasting Plastic Sales- Dataset
#Load Libraies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#Read the dataset
plastic=pd.read_csv('C:\\Data Science\\Assignments\\Forecasting\\PlasticSales.csv')
plastic.head()

#To make plotting graphs easier, we set the index of dataframe to the Month. During plots, it will be easier
plastic['Month'] = pd.to_datetime(plastic['Month'],infer_datetime_format=True)
indexed_dataset = plastic.set_index(['Month'])
indexed_dataset.head()


#print(type(plastic))
#print(type(indexed_dataset))
print(plastic.shape)

#Clean the data, check for any NUll values
plastic.isna().any()

#Statistical analysis
plastic.describe()    #Observations - check for any outliers, no outliers present

#Histogram Plot
plastic.hist()

#Plot Graph to check the trend, seasonality, etc.
plt.xlabel("Date")
plt.ylabel("Plastic Sales")
plt.plot(indexed_dataset)   #Observation - Trend is present and seasonality is also present. This is an upward trent

#Create Dummy Variable of t - values 1 to 60
plastic["t"] = np.arange(1,61)
plastic.tail()


# Create a Base Model/Naive Model
#Create a baseline Model/Naive Model
plastic_base = plastic[["Month","Sales"]]
plastic_base.head()

#Create a baseline Model/Naive Model
plastic_base = pd.concat([plastic_base, plastic_base['Sales'].shift(1)], axis=1)
plastic_base.head()

plastic_base.columns = ['Month','Actual_Sales', 'Forecast_Sales']
plastic_base.dropna(inplace=True)
plastic_base.head()

#Plot
plastic_base[['Actual_Sales','Forecast_Sales']].plot(figsize=(12,8))

print('Naive Model - MSE %d, RMSE %d', plastic_error, np.sqrt(plastic_error))   #MSE = 15722.15, RMSE Value - 125.388

indexed_dataset.head()

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
train = plastic[0:48]
test = plastic[48:60]
#print(train.shape)
#print(test.shape)
#print(test.index)
#print(train.index)

# Creating a functions to calculate RMSE and MAPE values 

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    pred = pred.astype(float)
    org = org.astype(float)
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)


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
print('Linear Model - RMSE %d, MAPE %d', rmse_linear, MAPE(pred_linear,test.Sales))
#RMSE - 260.93, MAPE = 18.96

#########################Simple Exponential Method###########################################
# Simple Exponential Method
ses_model = SimpleExpSmoothing(train["Sales"]).fit()
pred_ses = ses_model.predict(start = test.index[0],end = test.index[-1])
#MAPE = MAPE(pred_ses,test.Sales) # 17.04
#RMSE = RMSE(pred_ses,test.Sales) # 264.56
print('Simple Exp Model - RMSE %d, MAPE %d', RMSE(pred_ses,test.Sales), MAPE(pred_ses,test.Sales))

#########################Holt Method###########################################
# Holt method 
hw_model = Holt(train["Sales"]).fit()
pred_hw = hw_model.predict(start = test.index[0],end = test.index[-1])
#MAPE = MAPE(pred_hw,test.Sales) # 101.95
#RMSE = RMSE(pred_hw,test.Sales) # 1570.11
print('Holt Model - RMSE %d, MAPE %d', RMSE(pred_hw,test.Sales), MAPE(pred_hw,test.Sales))
##################################################################################

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(train["Sales"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = test.index[0],end = test.index[-1])
#MAPE(pred_hwe_add_add,test.Sales) # 14.42
#RMSE = RMSE(pred_hwe_add_add,test.Sales) # 214
print('Holts winter exponential smoothing - RMSE %d, MAPE %d', RMSE(pred_hwe_add_add,test.Sales), MAPE(pred_hwe_add_add,test.Sales))

########################################################################3
# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = test.index[0],end = test.index[-1])
#MAPE(pred_hwe_mul_add,test.Sales) # MAPE - 15.002, RMSE - 243.544
print('Holts winter ES - Mult n Add - RMSE %d, MAPE %d', RMSE(pred_hwe_mul_add,test.Sales), MAPE(pred_hwe_mul_add,test.Sales))

###########################Plot - Visualization ##################################
#Visualization of Forecasted values for Test data set using different methods 
plt.plot(train.index, train["Sales"], label='Train',color="black")
plt.plot(test.index, test["Sales"], label='Test',color="blue")
plt.plot(pred_linear.index, pred_linear, label='Linear',color="violet")
plt.plot(pred_ses.index, pred_ses, label='SimpleExponential',color="green")
plt.plot(pred_hw.index, pred_hw, label='Holts_winter',color="orange")
plt.plot(pred_hwe_add_add.index,pred_hwe_add_add,label="HoltsWinterExponentialAdd",color="brown")
plt.plot(pred_hwe_mul_add.index,pred_hwe_mul_add,label="HoltsWinterExponentialMul",color="grey")
plt.legend(loc='best')
#Observation
#HoltsWinterExponentialAdd is close to the prediction - except higher magnitude
##HoltsWinterExponentialMul has more magnitude, also gives average prediction but of higher magnitude

#####################################################################
### Testing For Stationarity - ADF (Augmented Dickey Fuller) Test
#####################################################################
from statsmodels.tsa.stattools import adfuller

#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(timeseries):
    result=adfuller(timeseries)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
        
adfuller_test(indexed_dataset['Sales'])


#Differencing Approach
plastic['Sales First Difference'] = plastic['Sales'] - plastic['Sales'].shift(1)
plastic['Sales First Difference'].dropna().plot()

adfuller_test(plastic['Sales First Difference'].dropna())


plastic['Seasonal First Difference']=plastic['Sales']-plastic['Sales'].shift(12)
plastic['Seasonal First Difference'].dropna().plot()
#plastic.head()

adfuller_test(plastic['Seasonal First Difference'].dropna())


##########################################################
#Auto ARIMA model works on the efficient Grid Search and Random Search concepts to find the most optimal parameters to find #the best fitting time series model.
#############################################################################
# Lets us use auto_arima from p
#pip install -q pyramid_arima - from Ananconda Command Prompt
import pyramid
from pyramid.arima import auto_arima

# Lets us use auto_arima from p
auto_arima_model = auto_arima(indexed_dataset,start_p=0,
                              start_q=0,max_p=10,max_q=10,
                              m=12,start_P=0,seasonal=True,
                              d=1,D=1,trace=True,error_action="ignore",
                              suppress_warnings= True,
                              stepwise=True)
            
       
auto_arima_model.summary()  # SARIMAX(0, 1, 0)x(0, 1, 1, 12)
# AIC ==> 511.396
# BIC ==> 516.946

#The best performing model from the optimized grid search is the following-
#ARIMA: order=(0, 1, 0) seasonal_order=(0, 1, 1, 12); AIC=511.396, BIC=516.946            
###############Output########################################################
#Fit ARIMA: order=(0, 1, 0) seasonal_order=(0, 1, 1, 12); AIC=511.396, BIC=516.946, Fit time=0.238 seconds
#Fit ARIMA: order=(0, 1, 0) seasonal_order=(0, 1, 0, 12); AIC=514.612, BIC=518.312, Fit time=0.015 seconds
#Fit ARIMA: order=(1, 1, 0) seasonal_order=(1, 1, 0, 12); AIC=515.419, BIC=522.820, Fit time=0.258 seconds
#Fit ARIMA: order=(0, 1, 1) seasonal_order=(0, 1, 1, 12); AIC=513.219, BIC=520.620, Fit time=0.423 seconds
#Fit ARIMA: order=(0, 1, 0) seasonal_order=(1, 1, 1, 12); AIC=513.199, BIC=520.599, Fit time=0.623 seconds
#Fit ARIMA: order=(0, 1, 0) seasonal_order=(0, 1, 2, 12); AIC=nan, BIC=nan, Fit time=nan seconds
#Fit ARIMA: order=(0, 1, 0) seasonal_order=(1, 1, 2, 12); AIC=nan, BIC=nan, Fit time=nan seconds
#Fit ARIMA: order=(1, 1, 0) seasonal_order=(0, 1, 1, 12); AIC=513.209, BIC=520.610, Fit time=0.419 seconds
#Fit ARIMA: order=(1, 1, 1) seasonal_order=(0, 1, 1, 12); AIC=513.581, BIC=522.831, Fit time=0.964 seconds
#Total fit time: 2.949 seconds
#Statespace Model Results
#Dep. Variable:	y	No. Observations:	60
#Model:	SARIMAX(0, 1, 0)x(0, 1, 1, 12)	Log Likelihood	-252.698
#Date:	Sun, 17 May 2020	AIC	511.396
#Time:	07:39:05	BIC	516.946
#Sample:	0	HQIC	513.484
#- 60		
#Covariance Type:	opg		
#coef	std err	z	P>|z|	[0.025	0.975]
#intercept	-2.6534	4.843	-0.548	0.584	-12.145	6.838
#ma.S.L12	-0.6079	0.275	-2.207	0.027	-1.148	-0.068
#sigma2	2437.7681	700.089	3.482	0.000	1065.620	3809.916
#Ljung-Box (Q):	25.14	Jarque-Bera (JB):	4.81
#Prob(Q):	0.97	Prob(JB):	0.09
#Heteroskedasticity (H):	4.99	Skew:	-0.78
#Prob(H) (two-sided):	0.00	Kurtosis:	2.82
#########################################################################

# Create predictions, evaluate on test
period = plastic.shape[0] + 12
preds, conf_int = auto_arima_model.predict(n_periods=period, return_conf_int=True)
#print(preds)
#print(conf_int)

#Plot to check prediction and forecast of the model
plt.figure(figsize=(15,10))
plt.plot(preds, color="orange",label="Forecast")
plt.plot(plastic['Sales'][0:60], color="blue", label="Actual")
plt.legend(loc="best")

preds1, conf_int1 = auto_arima_model.predict(n_periods=test.shape[0], return_conf_int=True)
RMSEval = np.sqrt(mean_squared_error(test['Sales'], preds1))
MAPEtemp = np.abs((preds1-test['Sales']))*100/test['Sales']    
MAPEtemp1 = MAPEtemp.mean()   #RMSE 164.43727069698656, MAPE - 11.31346953520431

print('RMSE %d, MAPE %d', RMSEval, MAPEtemp1)

#Observation - we need to make the series Stationary for the model to be valid

###############################################################################


#######################SARIMA Model#######################
#SARIMA Model
import statsmodels.api as sm
mod = sm.tsa.statespace.SARIMAX(indexed_dataset,
                                order=(0, 1, 0),
                                seasonal_order=(0, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

#############################################
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ma.S.L12   -1.507e+14         -0        inf      0.000   -1.51e+14   -1.51e+14
sigma2       6.18e-14   3.25e-10      0.000      1.000   -6.38e-10    6.38e-10
==============================================================================

#############################SARIMA###########################################

# For getting predictions for future we use predict() function 
SARIMApredict = results.predict(start=0,end=59,dynamic=False)
#SARIMApredict
# Forecast values
SARIMAforecast = results.forecast(steps=12)
#SARIMAforecast
#Combine together - Prediction + forecast
SARIMAdf = pd.concat([SARIMApredict, SARIMAforecast])
SARIMAdf.head()
Finaldf = pd.DataFrame(data=SARIMAdf, columns=["Sales"])
Finaldf


#Plot to check prediction and forecast of the model
plt.figure(figsize=(15,10))
plt.plot(SARIMAdf, color="orange",label="Forecast")
plt.plot(indexed_dataset, color="blue", label="Actual")
plt.legend(loc="best")

##############################TRANSFORMATIONS################################
#We need to make the series stationary to apply ARIMA Model
#Transformation
plastic['plastic_log'] = np.log(plastic['Sales'])
plastic['plastic_log_diff'] = plastic['plastic_log'] - plastic['plastic_log'].shift(1)
plastic['plastic_log_diff'].dropna().plot()

#Apply Dicky Fuller test
adfuller_test(plastic['plastic_log_diff'].dropna())    #p-value 0.81 , not-stationary

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

datasetLogScaleMinusMovingAverage = indexed_dataset_log - movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#Remove NAN values
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)

#Plot
datasetLogScaleMinusMovingAverage.dropna().plot()

#Apply Test
adfuller_test(datasetLogScaleMinusMovingAverage.iloc[:,0].values) #Observation - p-value 1.809, still it is non-stationary 

#Log Differencing
datasetLogDiffShifting = indexed_dataset_log - indexed_dataset_log.shift()
#Remove NAN values
datasetLogDiffShifting.dropna(inplace=True)

#Apply Test
adfuller_test(datasetLogDiffShifting.iloc[:,0].values)  #p-value 0.8162667125730252 - still it is non-stationary

#Apply Transformation
datasetLogScaleMinusExponentialMovingAverage = indexed_dataset_log - exponentialDecayWeightedAverage
adfuller_test(datasetLogScaleMinusExponentialMovingAverage.iloc[:,0].values)
#plt.plot(indexed_dataset_log)
plt.plot(datasetLogScaleMinusExponentialMovingAverage, color='red')
#Observation - p-value has decreased ~ 0, this is stationary

##############################################################
Eliminating Trend and Seasonality
#####################################################
#Step 1. Differencing
plt.plot(datasetLogDiffShifting.dropna())
adfuller_test(datasetLogDiffShifting.iloc[:,0].values)  #p-value : 0.8162667125730252

####################################################
####################################################
# Step 2- Decomposing technique
####################################
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
#decomposedLogData

adfuller_test(decomposedLogData.iloc[:,0].values)   # p-value 0.0008370244245030603 -  This is now stationary

########################################


#Plotting ACF & PACF plots to arrive at p and q values
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

plot_pacf(datasetLogDiffShifting)
plot_acf(datasetLogDiffShifting)

#Based on the plot - p-value from PACF = 2
#q-value from ACF - 2

#########################################################
# Build Models##########################################################
######AR Model ########################################
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
plt.title('RSS: %.4f'%sum((results_AR.fittedvalues - datasetLogDiffShifting['Sales'])**2))
print('Plotting AR model')
print('RMSE: %.4f'%np.sqrt(mean_squared_error(results_AR.fittedvalues, datasetLogDiffShifting['Sales'])))
#RSS - 0.4019, RMSE = 0.0825


#MA Model # ################################################
model = ARIMA(indexed_dataset_log, order=(0,1,1))
results_MA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_MA.fittedvalues - datasetLogDiffShifting['Sales'])**2))
print('Plotting MA model')
print('RMSE: %.4f'%np.sqrt(mean_squared_error(results_MA.fittedvalues, datasetLogDiffShifting['Sales'])))
#RSS: 0.4747
#RMSE: 0.0897

###################################################################
#ARIMA
############################################
# AR+I+MA = ARIMA model   #0.6931
model = ARIMA(indexed_dataset_log, order=(1,1,1))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - datasetLogDiffShifting['Sales'])**2))
print('Plotting ARIMA model')
print('RMSE: %.4f'%np.sqrt(mean_squared_error(results_ARIMA.fittedvalues, datasetLogDiffShifting['Sales'])))
#RMSE: 0.0835
#RSS:0.4112

#################################
#Best ARIMA Model Selection
# AR+I+MA = ARIMA model   Best Model AR Model-  RSS #0.4019, RMSE - 0.0825
model = ARIMA(indexed_dataset_log, order=(2,1,0))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - datasetLogDiffShifting['Sales'])**2))
print('Plotting ARIMA model')
print('RMSE: %.4f'%np.sqrt(mean_squared_error(results_ARIMA.fittedvalues, datasetLogDiffShifting['Sales'])))

results_ARIMA.summary()
results_ARIMA.aic # -118.71
#########################################
#Prediction & Reverse transformations
#############################################################
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head(12))

#Convert to cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum)

predictions_ARIMA_log = pd.Series(indexed_dataset_log['Sales'].iloc[0], index=indexed_dataset_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()

#################################
# Inverse of log is exp. Final
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(indexed_dataset, color="blue")
plt.plot(predictions_ARIMA, color="orange")

###############################################################################
#Sarima Model gives very good accuracy - replotting what was already calculated and forecasted in steps above
# For getting predictions for future we use predict() function 
#Plot to check prediction and forecast of the model
plt.figure(figsize=(15,10))
plt.plot(SARIMAdf, color="orange",label="Forecast")
plt.plot(indexed_dataset, color="blue", label="Actual")
#plt.fill_between(lower_series.index, lower_series, upper_series, 
#                 color='k', alpha=.15)
plt.legend(loc="best")