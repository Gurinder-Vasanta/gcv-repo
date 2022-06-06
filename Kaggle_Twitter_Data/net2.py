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

#to get the imports to work properly
#virtualenv ENV
#source ENV/bin/activate

#original version
loop_cap = 1000
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': Activation(swish)})

data = pd.read_csv('data.csv',encoding='latin-1')
targets = np.array(data['target'])
tweets = np.array(data['text'])
all_distinct_words = []
#v originally len(data)
#v this creates the all distinct words array
for i in range(loop_cap):
    split_text_arr = tweets[i].split(' ')
    #split_text_arr = split_text_arr[split_text_arr != '']
    for j in range(len(split_text_arr)):
        if(split_text_arr[j] not in all_distinct_words):
            all_distinct_words.append(split_text_arr[j])
    #print(i)
    '''print(tweets[i])
    print('<<<<<<<<<<<<< tweets[i] >>>>>>>>>>>>>>')
    print()
    print(split_text_arr)
    print('------------- split_text_arr -------------')
    print()
    print(all_distinct_words)
    input('code break')'''

all_distinct_words = np.array(all_distinct_words)
print(all_distinct_words)
print(len(all_distinct_words))
train_x = []
train_y = []
#v should actually be len(tweets)
#v this processes the tweets
for i in range(loop_cap):
    train_y.append(float(targets[i]))
    cache = []
    for j in range(len(all_distinct_words)):
        cache.append(0.0)
    for k in range(len(all_distinct_words)):
        if(all_distinct_words[k] in tweets[i]):
            cache[k] = 1.0
    train_x.append(cache)

train_x = np.array(train_x).reshape(loop_cap,1,len(all_distinct_words))
train_y = np.array(train_y).reshape(loop_cap,1,1)

print(np.ndarray.tolist(train_y))
input('stop')
#1
#2
#83
#163
#166
#326
#13529
#27058
model = Sequential()
model.add(LSTM(1304, activation='swish', return_sequences = True, input_shape = (1,len(all_distinct_words))))
model.add(LSTM(83, activation='swish', return_sequences = True))
model.add(Dense(1, activation = 'relu'))

learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(0.001)
optimizer.learning_rate.assign(learning_rate)

model.compile(optimizer=optimizer, loss='mse')

history = model.fit(train_x, train_y, epochs=3, verbose=1)

print(model.predict(train_x[998].reshape(1,1,4707)))
print(train_y[998])
print(train_x.shape)
print(train_y.shape)
#to get the imports to work properly
#virtualenv ENV
#source ENV/bin/activate


#there are a total of 1,350,487 distinct words/characters in all of the 1.6 million messages
