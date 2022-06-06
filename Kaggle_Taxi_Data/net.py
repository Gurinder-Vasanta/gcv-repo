import pandas as pd
import numpy as np
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
data = pd.read_csv('train.csv')
passenger_counts = data['passenger_count']
pickup_longitude = data['pickup_longitude']
pickup_latitude = data['pickup_latitude']
dropoff_longitude = data['dropoff_longitude']
dropoff_latitude = data['dropoff_latitude']
trip_time = data['trip_duration']
train_x = []
train_y = []
means = [np.mean(np.array(passenger_counts)),np.mean(np.array(pickup_latitude)),np.mean(np.array(pickup_longitude)),
         np.mean(np.array(dropoff_latitude)),np.mean(np.array(dropoff_longitude))]
stdevs = [np.std(np.array(passenger_counts)),np.std(np.array(pickup_latitude)),np.std(np.array(pickup_longitude)),
         np.std(np.array(dropoff_latitude)),np.std(np.array(dropoff_longitude))]
print(means)
print(stdevs)
for i in range(len(data)):
    temp = [passenger_counts[i],pickup_latitude[i],pickup_longitude[i],dropoff_latitude[i],dropoff_longitude[i]]
    train_x.append(temp)
train_x = np.array(train_x)
print(train_x)
print(train_x.shape)
def normalizer(x_array,means,stdevs):
    output = []

    for i in range(len(x_array)):
        num = x_array[i] - means[i]
        z_score = num/stdevs[i]
        output.append(z_score)
    return output

def denormalizer(z_score,mean,stdev):
    x = (z_score * stdev)+mean
    return x
for i in range(len(train_x)):
    train_x[i] = normalizer(train_x[i],means,stdevs)
print(train_x)
y_mean = np.mean(trip_time)
y_stdev = np.std(trip_time)
for i in range(len(trip_time)):
   num = trip_time[i] - y_mean
   z_score = num/y_stdev
   train_y.append(z_score)
train_y = np.array(train_y)

train_x = train_x.reshape(1458644,1,5)
train_y = train_y.reshape(1458644,1,1)
model = Sequential()
model.add((Dense(5*100, activation='tanh', input_shape = (1,5))))
#model.add(Dropout(0.025))
model.add((Dense(5*25, activation='tanh')))
model.add((Dense(5*5, activation='tanh')))
model.add((Dense(1, activation='tanh')))
#lr of 0.00005 is peak
#100 to 25 is peak
#100 to 25 to 5
learning_rate = 0.00005
optimizer = tf.keras.optimizers.Adam(0.001)
optimizer.learning_rate.assign(learning_rate)
#3 epochs
model.compile(optimizer=optimizer, loss='mae', metrics=['mse','acc'])
history = model.fit(train_x, train_y, epochs=10, verbose=1)
results = open('train_results10eps.csv','w')
results.write('prediction,actual,denormalized prediction,denormalized actual,%error\n')

for i in range(5000):

    print(str(model.predict(train_x[i].reshape(1,1,5))[0][0][0]) + '   <------ prediction       actual ------>   ' + str(train_y[i][0][0]))
    print(str(denormalizer(model.predict(train_x[i].reshape(1,1,5))[0][0][0],y_mean,y_stdev))+ '   <------ denormalized prediction      denormalized actual ------>   ' + str(denormalizer(train_y[i][0][0],y_mean,y_stdev)))
    num = abs(denormalizer(model.predict(train_x[i].reshape(1,1,5))[0][0][0],y_mean,y_stdev) - denormalizer(train_y[i][0][0],y_mean,y_stdev))
    frac = num/(denormalizer(train_y[i][0][0],y_mean,y_stdev))
    per_error = frac * 100
    results.write(str(model.predict(train_x[i].reshape(1, 1, 5))[0][0][0]) + ',' + str(train_y[i][0][0]) + ',' + str(denormalizer(model.predict(train_x[i].reshape(1, 1, 5))[0][0][0], y_mean, y_stdev)) + ',' + str(denormalizer(train_y[i][0][0], y_mean, y_stdev)) + ',' + str(per_error) + '\n')