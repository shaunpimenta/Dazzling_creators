
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
# import seaborn as sns
# import plotly.express as px
# from itertools import product
# import warnings
# import statsmodels.api as sm
plt.style.use('seaborn-darkgrid')


# In[57]:


# Reading the csv file
bitstamp = pd.read_csv("bitmap.csv")
bitstamp.head()


# In[59]:


bitstamp.info()


# In[60]:


bitstamp_non_indexed = bitstamp.copy()


# In[61]:


# Converting the Timestamp column from string to datetime
bitstamp['Timestamp'] = [datetime.fromtimestamp(x) for x in bitstamp['Timestamp']]


# In[62]:


bitstamp.head()


# In[ ]:





# In[63]:


print('Dataset Shape: ',  bitstamp.shape)


# In[64]:


bitstamp.set_index("Timestamp").Weighted_Price.plot(figsize=(14,7), title="Bitcoin Weighted Price")


# In[65]:


#calculating missing values in the dataset

missing_values = bitstamp.isnull().sum()
missing_per = (missing_values/bitstamp.shape[0])*100
missing_table = pd.concat([missing_values,missing_per], axis=1, ignore_index=True) 
missing_table.rename(columns={0:'Total Missing Values',1:'Missing %'}, inplace=True)
missing_table


# In[68]:



# bitstamp = bitstamp.set_index('Timestamp')
bitstamp.head()


# In[69]:


#testing missing value methods on a subset

pd.set_option('display.max_rows', 1500)

a = bitstamp.set_index('Timestamp')

a = a['2019-11-01 00:15:00':'2019-11-01 02:24:00']

a['ffill'] = a['Weighted_Price'].fillna(method='ffill') # Imputation using ffill/pad
a['bfill'] = a['Weighted_Price'].fillna(method='bfill') # Imputation using bfill/pad
a['interp'] = a['Weighted_Price'].interpolate()         # Imputation using interpolation

a


# In[70]:


def fill_missing(df):
    ### function to impute missing values using interpolation ###
    df['Open'] = df['Open'].interpolate()
    df['Close'] = df['Close'].interpolate()
    df['Weighted_Price'] = df['Weighted_Price'].interpolate()

    df['Volume_(BTC)'] = df['Volume_(BTC)'].interpolate()
    df['Volume_(Currency)'] = df['Volume_(Currency)'].interpolate()
    df['High'] = df['High'].interpolate()
    df['Low'] = df['Low'].interpolate()

    print(df.head())
    print(df.isnull().sum())


# In[71]:


fill_missing(bitstamp)


# In[72]:


#created a copy 
bitstamp_non_indexed = bitstamp.copy()


# In[73]:


bitstamp = bitstamp.set_index('Timestamp')
bitstamp.head()


# In[ ]:


ax = bitstamp['Weighted_Price'].plot(title='Bitcoin Prices', grid=True, figsize=(14,7))
ax.set_xlabel('Year')
ax.set_ylabel('Weighted Price')

ax.axvspan('2018-12-01','2019-01-31',color='red', alpha=0.3)
ax.axhspan(17500,20000, color='green',alpha=0.3)


# In[74]:


#Zooming in

ax = bitstamp.loc['2017-10':'2019-03','Weighted_Price'].plot(marker='o', linestyle='-',figsize=(15,6), title="Oct-17 to March-19 Trend", grid=True)
ax.set_xlabel('Month')
ax.set_ylabel('Weighted_Price')


# In[75]:


sns.kdeplot(bitstamp['Weighted_Price'], shade=True)


# In[76]:


# plt.figure(figsize=(15,12))
# plt.suptitle('Lag Plots', fontsize=22)
#
# plt.subplot(3,3,1)
# pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=1) #minute lag
# plt.title('1-Minute Lag')
#
# plt.subplot(3,3,2)
# pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=60) #hourley lag
# plt.title('1-Hour Lag')
#
# plt.subplot(3,3,3)
# pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=1440) #Daily lag
# plt.title('Daily Lag')
#
# plt.subplot(3,3,4)
# pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=10080) #weekly lag
# plt.title('Weekly Lag')
#
# plt.subplot(3,3,5)
# pd.plotting.lag_plot(bitstamp['Weighted_Price'], lag=43200) #month lag
# plt.title('1-Month Lag')
#
# plt.legend()
# plt.show()


# In[77]:


hourly_data = bitstamp.resample('1H').mean()
hourly_data = hourly_data.reset_index()

hourly_data.head()


# In[78]:


bitstamp_daily = bitstamp.resample("24H").mean() #daily resampling


# In[79]:


import plotly.express as px

bitstamp_daily.reset_index(inplace=True)
fig = px.line(bitstamp_daily, x='Timestamp', y='Weighted_Price', title='Weighted Price with Range Slider and Selectors')
fig.update_layout(hovermode="x")

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=2, label="2y", step="year", stepmode="backward"),
            dict(step="all")
            
        ])
    )
)
fig.show()


# In[80]:


plot_ = bitstamp_daily.set_index("Timestamp")["2017-12"]


# In[81]:


# import plotly.graph_objects as go
#
# fig = go.Figure(data=go.Candlestick(x= plot_.index,
#                     open=plot_['Open'],
#                     high=plot_['High'],
#                     low=plot_['Low'],
#                     close=plot_['Close']))
# fig.show()


# In[82]:


price_series = bitstamp_daily.reset_index().Weighted_Price.values
price_series


# In[83]:


# from statsmodels.tsa.seasonal import seasonal_decompose
# from statsmodels.tsa.stattools import kpss
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[84]:


fill_missing(bitstamp_daily)


# In[ ]:





# In[85]:

#
# plt.figure(figsize=(15,12))
# series = bitstamp_daily.Weighted_Price
# result = seasonal_decompose(series, model='additive',freq=1)
# result.plot()
#
#
# # In[86]:
#
#
# acf = plot_acf(series, lags=50, alpha=0.05)
# plt.title("ACF for Weighted Price", size=20)
# plt.show()
#
#
# # In[87]:
#
#
# plot_pacf(series, lags=50, alpha=0.05, method='ols')
# plt.title("PACF for Weighted Price", size=20)
# plt.show()


# In[88]:

#
# stats, p, lags, critical_values = kpss(series, 'ct')
#
#
# # In[89]:
#
#
# print(f'Test Statistics : {stats}')
# print(f'p-value : {p}')
# print(f'Critical Values : {critical_values}')
#
# if p < 0.05:
#     print('Series is not Stationary')
# else:
#     print('Series is Stationary')
#
#
# # In[90]:


def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    
    print (dfoutput)
    
    if p > 0.05:
        print('Series is not Stationary')
    else:
        print('Series is Stationary')
        
adf_test(series)


# In[91]:


df = bitstamp_daily.set_index("Timestamp")


# In[147]:


df.reset_index(drop=False, inplace=True)

lag_features = ["Open", "High", "Low", "Close","Volume_(BTC)"]
window1 = 3
window2 = 7
window3 = 30

df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)

df_mean_3d = df_rolled_3d.mean().shift(1).reset_index()
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index()
df_mean_30d = df_rolled_30d.mean().shift(1).reset_index()

df_std_3d = df_rolled_3d.std().shift(1).reset_index()
df_std_7d = df_rolled_7d.std().shift(1).reset_index()
df_std_30d = df_rolled_30d.std().shift(1).reset_index()

for feature in lag_features:
    df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
    df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
    df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]
    
    df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
    df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
    df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]

df.fillna(df.mean(), inplace=True)

df.set_index("Timestamp", drop=False, inplace=True)
df.head()


# In[152]:


df.to_csv('hello.csv')


# In[93]:


df["month"] = df.Timestamp.dt.month
df["week"] = df.Timestamp.dt.week
df["day"] = df.Timestamp.dt.day
df["day_of_week"] = df.Timestamp.dt.dayofweek
df.head()


# In[94]:


df_train = df[df.Timestamp < "2020"]
df_valid = df[df.Timestamp >= "2020"]

print('train shape :', df_train.shape)
print('validation shape :', df_valid.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[96]:





# In[95]:


price_series = bitstamp_daily.reset_index().Weighted_Price.values
price_series


# In[96]:


price_series.shape


# In[97]:


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0, 1))
price_series_scaled = scaler.fit_transform(price_series.reshape(-1,1))


# In[98]:


# price_series_scaled, price_series_scaled.shape


# In[99]:


train_data, test_data = price_series_scaled[0:2923], price_series_scaled[2923:]


# In[100]:


# test_data


# In[101]:


# train_data.shape, test_data.shape


# In[38]:


# X_train


# In[102]:


def windowed_dataset(series, time_step):
    dataX, dataY = [], []
    for i in range(len(series)- time_step-1):
        a = series[i : (i+time_step), 0]
        dataX.append(a)
        dataY.append(series[i+ time_step, 0])
        
    return np.array(dataX), np.array(dataY)


# In[172]:


X_train, y_train = windowed_dataset(train_data, time_step=100)
X_test, y_test = windowed_dataset(test_data, time_step=100)
# X_test.save('numpy',a)
from numpy import savetxt
savetxt('data.csv', X_test, delimiter=',')


# In[167]:


# X_train.shape, y_train.shape, X_test.shape, y_test.shape


# In[170]:


#reshape inputs to be [samples, timesteps, features] which is requred for LSTM

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_test.save('numpy',a)

print(X_train.shape) 
print(X_test.shape)


# In[169]:


print(y_train.shape) 
print(y_test.shape)


# In[107]:


#Create Stacked LSTM Model

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Dropout


# In[108]:


# Initialising the LSTM
# regressor = Sequential()
#
# # Adding the first LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
# regressor.add(Dropout(0.2))
#
# # Adding a second LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 50, return_sequences = True))
# regressor.add(Dropout(0.2))
#
# # Adding a third LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 50, return_sequences = True))
# regressor.add(Dropout(0.2))
#
# # Adding a fourth LSTM layer and some Dropout regularisation
# regressor.add(LSTM(units = 50))
# regressor.add(Dropout(0.2))
#
# # Adding the output layer
# regressor.add(Dense(units = 1))
#
# # Compiling the RNN
# regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[109]:


# regressor.summary()
#
#
# # In[110]:
#
#
# # Fitting the RNN to the Training set
# history = regressor.fit(X_train, y_train, validation_split=0.1, epochs = 50, batch_size = 32, verbose=1, shuffle=False)
#

# In[112]:


plt.figure(figsize=(16,7))
plt.plot(history.history["loss"], label= "train loss")
plt.plot(history.history["val_loss"], label= "validation loss")
plt.legend()


# In[113]:


#Lets do the prediction and performance checking

train_predict = regressor.predict(X_train)
test_predict = regressor.predict(X_test)
print(test_predict)


# In[114]:


#transformation to original form

y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
train_predict_inv = scaler.inverse_transform(train_predict)
test_predict_inv = scaler.inverse_transform(test_predict)
print(y_test_inv)


# In[165]:


# plt.figure(figsize=(16,7))
# plt.plot(y_train_inv.flatten(), marker='.', label="Actual")
# plt.plot(train_predict_inv.flatten(), 'r', marker='.', label="Predicted")
# plt.legend()
# plt.savefig(r"C:\Users\Shaun\Desktop\PBL Mineral\Prypto\Prypto_venv\static\predict",facecolor='white', transparent=True,  bbox_inches='tight')

# plt.close()


# In[116]:

#IFRSFGRSBNGHJSRNGJNSJRSNJGNRGNEK
plt.figure(figsize=(16,7))
plt.plot(y_test_inv.flatten(), marker='.', label="Actual")
plt.plot(test_predict_inv.flatten(), 'r', marker='.', label="Predicted")
plt.legend()
plt.savefig(r"C:\Users\Shaun\Desktop\PBL Mineral\Prypto\Prypto_venv\static\predict",facecolor='white', transparent=True,  bbox_inches='tight')


# In[117]:


# from sklearn.metrics import mean_absolute_error, mean_squared_error
#
# train_RMSE = np.sqrt(mean_squared_error(y_train, train_predict))
# test_RMSE = np.sqrt(mean_squared_error(y_test, test_predict))
# train_MAE = np.sqrt(mean_absolute_error(y_train, train_predict))
# test_MAE = np.sqrt(mean_absolute_error(y_test, test_predict))

#
# print(f"Train RMSE: {train_RMSE}")
# print(f"Train MAE: {train_MAE}")
#
# print(f"Test RMSE: {test_RMSE}")
# print(f"Test MAE: {test_MAE}")


# In[118]:


# test_data.shape


# In[119]:
#
#
# lookback = len(test_data) - 100
# x_input=test_data[lookback:].reshape(1,-1)
# x_input.shape
#
#
# # In[120]:
#
#
# x_input
#

# In[121]:


#
# lookback, len(test_data)
#
#
# # In[122]:
#
#
# temp_input=list(x_input)
# temp_input=temp_input[0].tolist()
# temp_input
#
#
# # In[123]:
#

# len(temp_input)


# In[124]:


# demonstrate prediction for next 100 days
from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = regressor.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = regressor.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

# print(lst_output)


# In[126]:


# len(price_series_scaled)


# In[127]:


# df_=price_series_scaled.tolist()
# df_.extend(lst_output)
# plt.plot(df_)


# In[128]:


# plt.figure(figsize=(14,7))
# df_invscaled=scaler.inverse_transform(df_).tolist()
# plt.plot(df_invscaled)
#

# In[129]:


# import pickle
#
#
# # manually set the parameters of the figure to and appropriate size
# plt.rcParams['figure.figsize'] = [14, 10]
#
# # loss_values = [ev['loss'] for ev in evaluations]
# # training_steps = [ev['global_step'] for ev in evaluations]
#
# # plt.scatter(x=training_steps, y=loss_values)
# # plt.xlabel('Training steps (Epochs = steps / 2)')
# # plt.ylabel('Loss (SSE)')
# # plt.show()
# pickle.dump(regressor, open('model.pkl','wb'))
# # Loading model to compare the results
# lstm = pickle.load(open('model.pkl','rb'))


# In[134]:

#
# import keras
# # from keras.models impor1t model
# # model=keras.models.save_model('lstm1.pkl',filepath='')
# # model = keras.models.load_model('lstm1.pkl')
# import tensorflow as tf
# localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
# regressor.save('lstm.h5', options=localhost_save_option)
#

# In[160]:


# model = keras.models.load_model('lstm.h5')
# da=pd.DataFrame(X_test)
# da
#

# In[143]:


# pred=model.predict(X_test)


# In[141]:


# train_predict_inv = scaler.inverse_transform(train_predict)
# test_predict_inv = scaler.inverse_transform(pred)
# print(test_predict_inv)


# In[158]:


# timestamp='2011-12-31'
# pred=df.loc[df['Timestamp'] == timestamp]
# print(pred)
#     # df.loc[(df['col1'] == value) & (df['col2'] < value)] multiple
# X_test, y_test = windowed_dataset(pred, time_step=100)
# print(X_test.shape)
# price=model.predict(X_test)

#
# # In[178]:
#
#
# data=pd.read_csv('data.csv')
# data.head()
#
#
# # In[174]:
#
#
# price=model.predict(data)
#
#
# # In[177]:
#
#
# # print(price)
# test_predict_inv = scaler.inverse_transform(price)
# print(test_predict_inv)
#
#
# # In[ ]:
#
#
#
#
