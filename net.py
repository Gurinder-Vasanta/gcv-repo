import numpy as np
import pandas as pd
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from torch.utils.data import Dataset, DataLoader, random_split
import math
import tensorflow as tf
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': Activation(swish)})

data = pd.read_csv('ticker_size.csv')
tickers = np.array(data['Ticker'])
sizes = np.array(data['Size'])
dictionary = {}
for i in range(len(tickers)):
    dictionary[tickers[i]] = sizes[i]
#print((dictionary))
valid_tickers = []
valid_sizes = []
valid_dictionary = {}
for i in range(len(tickers)):
    if(dictionary[tickers[i]]>=1967):
        valid_tickers.append(tickers[i])
        valid_sizes.append(sizes[i])
for i in range(len(valid_tickers)):
    valid_dictionary[valid_tickers[i]] = valid_sizes[i]
all_train_x = []
dow_data = pd.read_csv('/Users/gcvasanta/Desktop/Python/Stock_Predictor/Data/dowj.csv')
sp500_data = pd.read_csv('/Users/gcvasanta/Desktop/Python/Stock_Predictor/Data/sp500.csv')
nasdaq_data = pd.read_csv('/Users/gcvasanta/Desktop/Python/Stock_Predictor/Data/nasdaq.csv')

dow_data = np.array(dow_data[-1966:-1]).reshape(1965)
sp500_data = np.array(sp500_data[-1966:-1]).reshape(1965)
nasdaq_data = np.array(nasdaq_data[-1966:-1]).reshape(1965)

x_means = [np.mean(dow_data),np.mean(sp500_data),np.mean(nasdaq_data)]
x_stdevs = [np.std(dow_data),np.std(sp500_data),np.std(nasdaq_data)]
for i in range(len(dow_data)):
    temp = [dow_data[i],sp500_data[i],nasdaq_data[i]]
    all_train_x.append(temp)
temp_x=[]
for i in range(len(valid_tickers)):
    cache = []
    aux = np.array(pd.read_csv('/Users/gcvasanta/Desktop/Python/Stock_Predictor/Data/'+valid_tickers[i]+'.csv'))

    aux = aux.reshape(aux.shape[0])[-1966:-1]
    #print(aux)
    #input('stop')
    x_means.append(np.mean(aux))
    x_stdevs.append(np.std(aux))
    cache.append(aux)
    cache = np.array(cache).reshape(len(aux))
    temp_x.append(cache)
temp_x = np.transpose(temp_x)
#all_train_x = all_train_x.reshape(1965,1,3)
#all_train_x = all_train_x.reshape(1965,3)
for i in range(len(temp_x)):
    all_train_x[i] = (all_train_x[i]) + np.ndarray.tolist(temp_x[i])
all_train_x = np.array(all_train_x)
all_train_x = np.array(all_train_x).reshape(1965,210)
def normalizer(x_array,mean_array,standard_deviation_array):
    temp = []
    for i in range(len(x_array)):
        numerator = float(x_array[i]-mean_array[i])
        z_score = numerator/standard_deviation_array[i]
        temp.append(z_score)
    return(temp)
def denormalizer(z_array,mean_array,standard_deviation_array):
    temp = []
    for i in range(len(z_array)):
        x = (z_array[i] * standard_deviation_array[i])+mean_array[i]
        temp.append(x)

    return temp
for i in range(len(all_train_x)):
    all_train_x[i] = normalizer(all_train_x[i],x_means,x_stdevs)

y_means = []
y_stdevs = []
train_y = []
print(all_train_x)
for i in range(len(valid_tickers)):
    cache = []
    aux = np.array(pd.read_csv('/Users/gcvasanta/Desktop/Python/Stock_Predictor/Data/'+valid_tickers[i]+'.csv'))
    aux = aux.reshape(aux.shape[0])[-1965::]
    y_means.append(np.mean(aux))
    y_stdevs.append(np.std(aux))
    cache.append(aux)
    cache = np.array(cache).reshape(len(aux))
    train_y.append(cache)
train_y = np.array(train_y)
train_y = np.transpose(train_y)
for i in range(len(train_y)):
    train_y[i] = np.array(normalizer(train_y[i],y_means,y_stdevs))
print(all_train_x.shape)
print(train_y.shape)
all_train_x = all_train_x.reshape(1965,1,210)
train_y = train_y.reshape(1965,1,207)
model = Sequential()
#4320 --> 2700 --> 1161 --> 387 dense
#8640 --> 7500 --> 3483
#387*9 --> 387 lr of 0.005 worked the best so far
model.add((LSTM(207*9, activation='PReLU', return_sequences = True, input_shape = (1,210))))
#model.add((LSTM(4320, activation='swish', return_sequences=True)))
#model.add(Dropout(0.2))
#model.add((LSTM(207*6, activation='PReLU', return_sequences=True)))
model.add(Dropout(0.2))
model.add((LSTM(207*3, activation='PReLU', return_sequences=True)))

model.add(Dense(207, activation = 'PReLU'))
#model.add(Dense(207*2, activation = 'PReLU'))

learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(0.001)
optimizer.learning_rate.assign(learning_rate)
#local 8 is 15 9; local 9 is 15 12
model.compile(optimizer=optimizer, loss='mse', metrics=['mae','acc'])
history = model.fit(all_train_x, train_y, epochs=50, verbose=1)
print(train_y[1964])
print(model.predict(all_train_x[1964].reshape(1,1,210)))
print(denormalizer(all_train_x[1964].reshape(210),x_means,x_stdevs))
print(denormalizer(train_y[1964].reshape(207),y_means,y_stdevs))
predictions = np.array(denormalizer(model.predict(all_train_x[1964].reshape(1,1,210)),y_means,y_stdevs)).reshape(207)
print(predictions)
#print(train_y)