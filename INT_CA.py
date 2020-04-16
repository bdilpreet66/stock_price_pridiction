# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 10:00:45 2020

@author: dilpreet
"""


from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, LeakyReLU, BatchNormalization, Dropout
import xgboost as xgb

import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_squared_error, r2_score


def parser(x):
    x = str(x)
    return datetime.datetime.strptime(x,'%b %d, %Y')

dataset = pd.read_csv("SBI (SBI) Historical Prices.csv", header=0)

dataset['Date'] = dataset['Date'][dataset['Date'].isna() == False].apply(parser)

dataset = dataset[:-2]
dataset.sort_values(by=['Date'], axis=0,inplace=True)

dataset['Price'] = dataset['Price'].astype(np.float64, copy=False)

plt.figure(figsize=(14, 5), dpi=100)
plt.plot(dataset['Date'], dataset['Price'], label='SBI stock')
plt.vlines(datetime.date(2018,12, 1), 0, 400, linestyles='--', colors='gray', label='Train/Test data cut-off')
plt.xlabel('Date')
plt.ylabel('INR')
plt.title('Figure 2: SBI stock price')
plt.legend()
plt.show()

num_training_days = int(dataset.shape[0]*.7)


def get_technical_indicators(dataset):
    # Create Momentum
    dataset['momentum'] = dataset['Price'] - 1
    
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Price'].rolling(window=7,min_periods=1).mean()
    dataset['ma21'] = dataset['Price'].rolling(window=21,min_periods=1).mean()
    
    # Create MACD
    dataset['26ema'] = dataset['Price'].rolling(window=26,min_periods=1).mean()
    dataset['12ema'] = dataset['Price'].rolling(window=12,min_periods=1).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])

    # Create Bollinger Bands
    dataset['20sd'] = dataset['Price'].rolling(window=20,min_periods=1).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset['Price'].ewm(com=0.5).mean()
    
    return dataset

dataset = get_technical_indicators(dataset)

def strip(x):
    x = x.replace('M', '')
    x = x.replace('K', '')
    return float(x)

dataset['Vol.'] = dataset['Vol.'].apply(strip)

def plot_technical_indicators(dataset, last_days):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0-last_days
    
    dataset = dataset.iloc[-last_days:, :]
    x_ = range(3, dataset.shape[0])
    x_ =list(dataset.index)
    
    plt.plot(dataset['ma7'],label='MA 7', color='g',linestyle='--')
    plt.plot(dataset['Price'],label='Closing Price', color='b')
    plt.plot(dataset['ma21'],label='MA 21', color='r',linestyle='--')
    plt.plot(dataset['upper_band'],label='Upper Band', color='c')
    plt.plot(dataset['lower_band'],label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for SBI - last {} days.'.format(last_days))
    plt.ylabel('USD')
    plt.legend()

    plt.legend()
    plt.show()

plot_technical_indicators(dataset, 1125)


# Fourier transforms to find local and global trends
dataset_FT = dataset[['Date','Price']]

close_fft = np.fft.fft(np.asarray(dataset_FT['Price'].tolist()))
fft_df = pd.DataFrame({'fft':close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

plt.figure(figsize=(14, 7), dpi=100)
fft_list = np.asarray(fft_df['fft'].tolist())
for num_ in [3, 6, 9, 100,500]:
    fft_list_m10= np.copy(fft_list); fft_list_m10[num_:-num_]=0
    plt.plot(np.fft.ifft(fft_list_m10), label='Fourier transform with {} components'.format(num_))
plt.plot(dataset_FT['Price'],  label='Real')
plt.xlabel('Days')
plt.ylabel('INR')
plt.title('Figure 3: SBI (close) stock prices & Fourier transforms')
plt.legend()
plt.show()


from collections import deque
items = deque(np.asarray(fft_df['absolute'].tolist()))
items.rotate(int(np.floor(len(fft_df)/2)))
plt.figure(figsize=(10, 7), dpi=80)
plt.stem(items)
plt.title('Figure 4: Components of Fourier transforms')
plt.show()

dataset['fft'] = fft_df['fft']
dataset['absolute'] = fft_df['absolute']
dataset['angle'] = fft_df['angle']

# ARIMA 
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot

series = dataset_FT['Price']
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)


plt.figure(figsize=(12, 6), dpi=100)
plt.plot(test, label='Real')
plt.plot(predictions, color='red', label='Predicted')
plt.xlabel('Days')
plt.ylabel('INR')
plt.title('Figure 5: ARIMA model on SBI stock')
plt.legend()
plt.show()

dataset['ARIMA'] = history

dataset = dataset.astype({"Open":np.float64,"High":np.float64,"Low":np.float64,"fft":np.float64})

# XGBoost - Feature extraction
def get_feature_importance_data(data):
    data = data.copy()
    y = data['Price']
    X = data.iloc[:, 1:]
    
    train_samples = int(X.shape[0] * 0.65)
 
    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:]
    
    return (X_train, y_train), (X_test, y_test)

(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = get_feature_importance_data(dataset.loc[:,['Price','Open','High','Low','Vol.','Change %']])

regressor = xgb.XGBRegressor(gamma=0.0,n_estimators=150,base_score=0.7,colsample_bytree=1,learning_rate=0.05)


xgbModel = regressor.fit(X_train_FI,y_train_FI, eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)], verbose=False)

eval_result = regressor.evals_result()

training_rounds = range(len(eval_result['validation_0']['rmse']))

plt.scatter(x=training_rounds,y=eval_result['validation_0']['rmse'],label='Training Error')
plt.scatter(x=training_rounds,y=eval_result['validation_1']['rmse'],label='Validation Error')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.title('Training Vs Validation Error')
plt.legend()
plt.show()


fig = plt.figure(figsize=(8,8))
plt.xticks(rotation='vertical')
plt.bar([i for i in range(len(xgbModel.feature_importances_))], xgbModel.feature_importances_.tolist(), tick_label=X_test_FI.columns)
plt.title('Figure 6: Feature importance of the technical indicators.')
plt.show()







class model_LSTM(object):
    def __init__(self):
        self.G = None
        self.AM = None
        
    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        self.G.add(LSTM(units=50, return_sequences=True, input_shape=(60,13)))
        self.G.add(Dropout(0.2))
        self.G.add(LSTM(units=150, return_sequences=True))
        self.G.add(Dropout(0.2))
        self.G.add(LSTM(units=150, return_sequences=True))
        self.G.add(Dropout(0.2))
        self.G.add(LSTM(units=50))
        self.G.add(Dropout(0.2))
        self.G.add(Dense(1))
        
        return self.G
        
    def LSTM_Model(self):
        if self.AM:
            return self.AM
        self.AM = Sequential()
        
        self.AM.add(self.generator())
        self.AM.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        
        return self.AM




class Train_LSTM(object):
    def __init__(self,Dataset):
        self.data = Dataset
        
        self.lstm = model_LSTM()
        
        self.LSTM_Model = self.lstm.LSTM_Model()
        
        self.train = None
        self.test = None
        
    def gettestbatch(self,data,batch_size=60):
        arr = np.zeros((588-batch_size,batch_size,13))
        y_train = []
        for i in range(batch_size, 588):
            arr[i-batch_size] = data[i-batch_size:i,[0,1,2,3,4,5,6,7,11,13,14,15,19]]
            try:
                y_train.append(data[i+1,0])
            except :
                pass
        y_train = np.array(y_train)
        return arr,y_train
    
    def getbatch(self,data,batch_size=60):
        arr = np.zeros((1108-batch_size,batch_size,13))
        y_train = []
        for i in range(batch_size, 1108):
            arr[i-batch_size] = data[i-batch_size:i,[0,1,2,3,4,5,6,7,11,13,14,15,19]]
            y_train.append(data[i,0])
        y_train = np.array(y_train)
        return arr,y_train
        
    def Process_Data(self,data,batch_size):
        sc = MinMaxScaler()
        data = sc.fit_transform(data)
        self.train, self.test = data[:1125], data[1125-batch_size:]
        
        
    def Train(self, batch_size=60):
        self.Process_Data(self.data.iloc[1:,:],batch_size)
        inputs,target = self.getbatch(self.train,batch_size)
        self.LSTM_Model.fit(inputs, target, epochs=150, batch_size=32)
        
        
        
    def Test(self,batch_size=60):
        inputs,actual_price = self.gettestbatch(self.test,batch_size)
        pred_val = self.LSTM_Model.predict(inputs)
        return actual_price, pred_val
        





model = Train_LSTM(dataset.iloc[:,1:])
model.Train(batch_size=60)
y_test,y_pred = gan.Test(batch_size=60)


plt.figure(figsize=(12, 6), dpi=100)
plt.plot(y_test, label='Real')
plt.plot(y_pred.T[0], color='red', label='Predicted')
plt.xlabel('Days')
plt.ylabel('INR')
plt.title('model on SBI stock')
plt.legend()
plt.show()

r2_score(y_test,y_pred)











