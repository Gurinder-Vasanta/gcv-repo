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
train_results = open('train_results_700_4.csv','w')
test_results = open('test_results_700_4.csv','w')

train_results.write('predicted,actual\n')
test_results.write('ImageID,Raw Classification,Rounded Classification\n')
epochs = 700
train_results.write('Epochs: ' + str(epochs) + '\n')

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': Activation(swish)})

all_train_data = pd.read_csv('train.csv')
test_x = np.array(pd.read_csv('test.csv'))

train_y = np.array(all_train_data['label'])

all_train_data = np.array(all_train_data)

train_x = []
for i in range(len(all_train_data)):
    cache = []
    for j in range(1,785):
        cache.append(all_train_data[i][j])
    train_x.append(cache)

train_x = np.array(train_x).reshape(42000, 1, 784)
train_y = np.array(train_y).reshape(42000, 1, 1)

test_x = test_x.reshape(28000, 1, 784)

model = Sequential()
model.add((LSTM(420, activation='swish', return_sequences = True, input_shape = (1,784))))
model.add((LSTM(60, activation='swish', return_sequences=True)))
model.add((LSTM(30, activation='relu', return_sequences=True)))
#if the network doesnt work like it should, remove the activation = 'relu' part from the dense layer,
#rerun the program, and then add the activation = 'relu' part again and run it
model.add(Dense(1, activation = 'relu'))
#700_3: lr of 0.00005
#700_4: lr of 0.0005
#1680 to 30 was good; 18/20 accuracy
#lr of 0.005 was best so far
#0.001
#0.005
#5 epochs 0.001 lr 420 to 60 both normal LSTM
#5 epochs; 0.001 lr; 420 layer1; 140 layer2;
#5 epochs; 0.001 lr; 420 layer1; 30 layer2; also works fine
#lr: 0.0005
learning_rate = 0.0005
optimizer = tf.keras.optimizers.Adam(0.001)
optimizer.learning_rate.assign(learning_rate)
#50 epochs was best
#after 650 epochs the loss basically remains around .0250; just max out there
model.compile(optimizer=optimizer, loss='mse', metrics=['mae','acc'])

history = model.fit(train_x, train_y, epochs=epochs, verbose=1)

counter = 0
for i in range(0,len(train_x)):
    train_results.write(str(model.predict(np.array(train_x[i]).reshape(1,1,784))[0][0][0]) + ',' + str(train_y[i][0][0]) + '\n')
    if(round(model.predict(np.array(train_x[i]).reshape(1,1,784))[0][0][0]) == float(train_y[i][0][0])):
        counter += 1
train_results.write('Accuracy: ' + str(counter) + '/' + str(len(train_x)) + '\n')

for i in range(0,len(test_x)):
    test_results.write(str(i+1) + ',' + str(model.predict(np.array(test_x[i]).reshape(1,1,784))[0][0][0]) + ',' + str(round(model.predict(np.array(test_x[i]).reshape(1,1,784))[0][0][0])) + '\n')
