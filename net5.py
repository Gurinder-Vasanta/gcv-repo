import tensorflow as tf
import PIL.Image
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import normalize
import random

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': Activation(swish)})

normal_dir = 'chest_xray/train/NORMAL'
pneumonia_dir = 'chest_xray/train/PNEUMONIA'
train_x = []
train_y = []

d3 = 0
d4 = 0
sizes_2 = []
normal = []

for filename in os.listdir(normal_dir):
    pic = PIL.Image.open('chest_xray/train/NORMAL/'+filename)
    pic_rgb = pic.convert("L")
    size = np.array(pic_rgb.size)
    sizes_2.append(size)
    #print(size)
    if(size[0] > d3):
        d3 = size[0]
    if(size[1] > d4):
        d4 = size[1]
    cache = np.array(pic_rgb)
    #print(size)
    #print(cache)
    cache = cache.reshape(size[0],size[1])
    normal.append(cache)
#40 to #250 are the indices where the values are correct

for i in range(len(normal)):
    normal[i] = normal[i].reshape(sizes_2[i][0],sizes_2[i][1],1)
    temp = tf.image.pad_to_bounding_box(normal[i],0,0,d3,d4)
    normal[i] = temp.numpy().reshape(d3,d4)


d1 = 0
d2 = 0
sizes = []
pneumonia = []
for filename in os.listdir(pneumonia_dir):
    pic = PIL.Image.open('chest_xray/train/PNEUMONIA/'+filename)
    pic_rgb = pic.convert("L")
    size = np.array(pic_rgb.size)
    sizes.append(size)
    #print(size)
    if(size[0] > d1):
        d1 = size[0]
    if(size[1] > d2):
        d2 = size[1]
    cache = np.array(pic_rgb)
    #print(size)
    #print(cache)
    cache = cache.reshape(size[0],size[1])
    pneumonia.append(cache)

for i in range(len(pneumonia)):
    pneumonia[i] = pneumonia[i].reshape(sizes[i][0],sizes[i][1],1)
    temp = tf.image.pad_to_bounding_box(pneumonia[i],0,0,d3,d4)
    pneumonia[i] = temp.numpy().reshape(d3,d4)

#factrs of 1341: 1,3,9,149,447,1341
#factors of 3875: 1,5,25,31,125,155,775,3875
#1341 * .333333 = 447
#3875 * .2 =  775
#len(normal) = 1341
#len(pneumonia) = 3875
t_num = 500
normal_indices = random.sample(range(0,1341),t_num)
pneumonia_indices = random.sample(range(0,3875),t_num)
#[5]*2663 syntax to add array of length 2663 filled with 5s
print(len(normal))
print(len(pneumonia))
for i in range(len(normal_indices)):
    train_x.append(normalize(normal[normal_indices[i]]))
    train_y.append(0)
#had to do len(normal) for pneumonia as well because of memory issues
for i in range(len(pneumonia_indices)):
    train_x.append(normalize(pneumonia[pneumonia_indices[i]]))
    train_y.append(1)
print(np.array(train_x).shape)
print(np.array(train_y).shape)
train_x = np.array(train_x).reshape(t_num*2,1,2663*2916)
train_y = np.array(train_y).reshape(t_num*2,1,1)
print(np.array(train_x).shape)
print(np.array(train_y).shape)
#print(train_x[0])
#print(train_y[0])
#train_x = np.array(train_x).reshape(t_num*2,2916,2663)
#train_y = np.array(train_y).reshape(t_num*2,2916,2663)

model = Sequential()
#originally 2916 to 3
model.add(LSTM(75, activation='tanh', return_sequences=True, input_shape = (1,2916*2663)))
#model.add(LSTM(3000, activation='swish', return_sequences=True, input_shape=(2916,2663)))
model.add(LSTM(15, activation='tanh', return_sequences=True))
model.add(Dense(1, activation='relu'))

#1 layer; 1000; lr of 0.000005; loss went down almost 0.17 per epoch; went to nan somewhere mid of epoch 9 out of 10;
#for config above, trained with y as 2916 by 2663 array
#pleateud at epoch four of 10 for this config:
#layer 1: 750
#lr: 0.0001
#epochs: 10
#appended 1 and 2 to train y
learning_rate = 0.00005
optimizer = tf.keras.optimizers.Adam(0.001)
optimizer.learning_rate.assign(learning_rate)

model.compile(optimizer=optimizer, loss='mse')

history = model.fit(train_x, train_y, epochs=1, verbose=1)

output = np.ndarray.tolist((model.predict(np.array(train_x[0]).reshape(1,1,2916*2663))).reshape(2916))
output1 = np.ndarray.tolist((model.predict(np.array(train_x[800]).reshape(1,1,2916*2663))).reshape(2916))
#output = np.ndarray.tolist(np.array(output))
#output1 = np.ndarray.tolist(np.array(output1))
print(output)
print(output1)
#print(output)
#print()
#print(output1)
#print(np.array(output).shape)
#print(np.array(output1).shape)
'''normal_file = open('normal_results.txt','w')
pneumonia_file = open('pneumonia_results.txt','w')
for i in range(len(normal)):
    temp = np.ndarray.tolist((model.predict(np.array(normal[i]).reshape(1,2916,2663))).reshape(2916))
    normal_file.write(str(temp[30:300]))
    normal_file.write('\n')
    #print(temp[100])
for i in range(len(pneumonia)):
    aux = np.ndarray.tolist((model.predict(np.array(pneumonia[i]).reshape(1,2916,2663))).reshape(2916))
    pneumonia_file.write(str(aux[30:300]))
    pneumonia_file.write('\n')'''
    #print(aux[100])
#print(output[40:250])
#print(output1[40:250])
'''for i in range(len(pneumonia)):
    train_x = np.append(tra)'''