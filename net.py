import numpy as np
import pandas as pd
import PIL.Image
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import normalize
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dropout
import random
import keras
from sklearn import preprocessing
#apparel size: 32226
#artwork size: 4957
#cars size: 8144
#dishes size: 5831
#furniture size: 10488
#illustrations size: 3347
#landmark size: 33063
#meme size: 3301
#packaged size: 23413
#storefronts size: 5387
#toys size: 2402
def normalizer(x_array,means,stdevs):
    z_array = []
    for i in range(len(x_array)):
        z = (x_array[i]-means[i])/stdevs[i]
        z_array.append(z)
    return np.array(z_array)
def funct(x):
    return (k.log((0+tf.math.exp(0.225*x))/2))
get_custom_objects().update({'funct': Activation(funct)})
labels = ['apparel','artwork','cars','dishes','furniture','illustrations','landmark','meme','packaged','storefronts','toys']
label_map = {}
for i in range(len(labels)):
    aux = []
    for j in range(len(labels)):
        if(labels[i] == labels[j]):
            aux.append(1)
        else:
            aux.append(0)
    label_map[labels[i]] = aux
print(label_map)
'''for i in range(len(labels)):
    label_map[labels[i]] = i+1'''
train_x = []
train_y = []
test_x = []
test_y = []
for i in range(len(labels)):
    temp = 0
    train_indices = random.sample(range(0,len(os.listdir(labels[i]))),2400)
    print(train_indices)
    for filename in os.listdir(labels[i]):
        print(labels[i])
        if(temp in train_indices):
            print(temp)
            pic = PIL.Image.open(labels[i] + '/' + filename)
            pic_rgb = np.array(pic.convert('L')).reshape(1,512*512)
            train_x.append(pic_rgb)
            train_y.append(np.array(label_map[labels[i]]).reshape(1,11))
            temp+=1
        else:
            print(temp)
            pic1 = PIL.Image.open(labels[i] + '/' + filename)
            pic_rgb1 = np.array(pic1.convert('L')).reshape(1,512*512)
            test_x.append(pic_rgb1)
            test_y.append(np.array(label_map[labels[i]]).reshape(1,11))
            temp+=1
'''for i in range(len(labels)):
    temp = 0
    for filename in os.listdir(labels[i]):
        print(labels[i])
        pic = PIL.Image.open(labels[i] + '/'+filename)
        pic_rgb = np.array(pic.convert("L")).reshape(1,512*512)
        print(pic_rgb)
        train_x.append(pic_rgb)
        temp += 1
        print(temp)
        train_y.append(np.array(label_map[labels[i]]).reshape(1,11))'''
train_x = np.array(train_x)
train_y = np.array(train_y)
print(train_x)
print(train_y)
print(np.shape(train_x))
print(np.shape(train_y))
temp = np.transpose(train_x)
scaler = preprocessing.MinMaxScaler(feature_range = (1,2))


#print(np.ndarray.tolist(preprocessing.normalize((train_x[2][0]).reshape(1,len(train_x[3][0])))))
print(pd.Series((preprocessing.normalize((train_x[2][0]).reshape(1,len(train_x[3][0])))).reshape(len(train_x[3][0])))*1000.0)
print((train_x[2][0]))
print(train_x[1][0])
#input('stop')
for i in range(len(train_x)):
    train_x[i][0] = pd.Series((preprocessing.normalize((train_x[i][0]).reshape(1,len(train_x[i][0])))).reshape(len(train_x[i][0])))*1000.0
    print(train_x[i][0])
print(train_x)
#input('stop')
'''temp_normalized = []
means = []
stdevs = []
for i in range(len(temp)):
    means.append(np.mean(temp[i]))
    stdevs.append(np.std(temp[i]))
print(means)
input('means above')
print(stdevs)
input('stdevs above')
print(train_x[0][0])
print(train_x[1][0])
input('stop')
for i in range(len(train_x)):
    train_x[i][0] = normalizer(train_x[i][0],means,stdevs)
print(train_x)
input('stop')'''
#220 to 110 to 55 to 11: 3232
model = Sequential()
#2220 to 550 to 11 worked best
#model.add(Dense(2220, activation='tanh', input_shape = (1,512*512)))
#550 to 110
#model.add(Dense(300, return_sequences = True, activation='tanh', input_shape = (1,512*512))) #-------> lr 0.005 originally tanh
#model.add(Dense(1110, activation='LeakyReLU'))
#model.add(LSTM(110, return_sequences = True, activation='relu'))

#264*10 to 66*5 to 11
model.add(Dense(264*10, activation='LeakyReLU',input_shape = (1,512*512))) #---------> originally relu
#model.add(keras.layers.BatchNormalization())
#model.add(Dense(110, activation='tanh', input_shape = (1,512*512)))

model.add(keras.layers.BatchNormalization()) 
model.add(Dense(66*5, activation='tanh'))
#model.add(keras.layers.BatchNormalization())
#model.add(Dropout(0.1))
model.add(Dense(11, activation='softmax')) #--------->
#0.0005
learning_rate = 0.005
optimizer = tf.keras.optimizers.Adam(0.001)
optimizer.learning_rate.assign(learning_rate)
model.compile(optimizer=Adadelta(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['mae','mse',tf.keras.metrics.CategoricalAccuracy()])
history = model.fit(train_x, train_y, epochs=10, verbose=1)

print(model.predict(np.array(train_x[0]).reshape(1,1,512*512)))
print(train_y[0])
print(model.predict(np.array(train_x[2400]).reshape(1,1,512*512)))
print(train_y[2400])
print(model.predict(np.array(train_x[4800]).reshape(1,1,512*512)))
print(train_y[4800])
print(model.predict(np.array(train_x[7200]).reshape(1,1,512*512)))
print(train_y[7200])
print(model.predict(np.array(train_x[9600]).reshape(1,1,512*512)))
print(train_y[9600])
print(model.predict(np.array(train_x[12000]).reshape(1,1,512*512)))
print(train_y[12000])
print(model.predict(np.array(train_x[14400]).reshape(1,1,512*512)))
print(train_y[14400])
print(model.predict(np.array(train_x[16800]).reshape(1,1,512*512)))
print(train_y[16800])
print(model.predict(np.array(train_x[19200]).reshape(1,1,512*512)))
print(train_y[19200])
print(model.predict(np.array(train_x[21600]).reshape(1,1,512*512)))
print(train_y[21600])
print(model.predict(np.array(train_x[24000]).reshape(1,1,512*512)))
print(train_y[24000])
print(model.predict(np.array(test_x[0]).reshape(1,1,512*512)))
print(test_y[0])
print(model.predict(np.array(test_x[100000]).reshape(1,1,512*512)))
print(test_y[100000])
print(model.predict(np.array(test_x[50000]).reshape(1,1,512*512)))
print(test_y[50000])