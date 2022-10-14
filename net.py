import PIL.Image
import os
import numpy as np
import tensorflow
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Masking
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

normal_dir = 'chest_xray/train/NORMAL'
pneumonia_dir = 'chest_xray/train/PNEUMONIA'
train_x = []
train_y = []
d1 = 0
d2 = 0
sizes = []
for filename in os.listdir(normal_dir):
    pic = PIL.Image.open('chest_xray/train/NORMAL/'+filename)
    size = np.array(pic.size)
    pic_rgb = pic.convert("RGB")
    sizes.append(size)
    #print(size)
    if(size[0] > d1):
        d1 = size[0]
    if(size[1] > d2):
        d2 = size[1]
    cache = np.array(pic)

    #print(cache)
    cache = cache.reshape(size[0],size[1])
    #print(cache.shape)
    #input('code break')
    '''for i in range((size[0])):
        for j in range((size[1])):
            cache.append(np.sum(np.array(pic_rgb.getpixel((i,j))))/3)
            print(np.sum(np.array(pic_rgb.getpixel((i,j))))/3)'''
    train_x.append(cache)
sizes = np.array(sizes)
for i in range(len(sizes)):
    print('in normal sizes')
    print(i)
#print(len(train_x))
#print(len(train_y))
#print(d1)
#print(d2)
'''for i in range(len(sizes)):
    print(sizes[i])'''
#print(train_x[0][0])
#print(sizes[0])
#print(len(train_x[0]))
max_rows = 2916
max_columns = 2663
masked_value = -1
#v actually supposed to be len(train_x)
actual_train = []
for i in range(2):
    train_x[i] = np.ndarray.tolist(train_x[i])
    arr = [-1] * 2663
    arr = np.array(arr)
    #print(sizes[i])
    #print(i)
    #print(sizes[i])
    for j in range(len(train_x[i])):
        while(len(train_x[i][j]) < 2663):
            train_x[i][j].append(-1)
    while(len(train_x[i])<2916):
        train_x[i].append(arr)
    train_x[i] = np.array(train_x[i])
    actual_train.append(train_x[i])
    train_y.append(0)
ptx = []
pty = []
psizes = []
'''for filename in os.listdir(pneumonia_dir):
    pic = PIL.Image.open('chest_xray/train/PNEUMONIA/' + filename)
    size = list(pic.size)
    pic_rgb = pic.convert("RGB")
    psizes = list(psizes)
    psizes.append(size)
    print(size)
    if (size[0] > d1):
        d1 = size[0]
    if (size[1] > d2):
        d2 = size[1]
    print(len(psizes))
    input('code break')
    cache = np.array(pic)
    print(cache.shape)
    # print(cache)
    cache = cache.reshape(size[0], size[1])
    # print(cache.shape)
    # input('code break')
    for i in range((size[0])):
        for j in range((size[1])):
            cache.append(np.sum(np.array(pic_rgb.getpixel((i,j))))/3)
            print(np.sum(np.array(pic_rgb.getpixel((i,j))))/3)
    ptx.append(cache)
    pty.append(0)
    psizes = np.array(psizes)
    # print(len(train_x))
    # print(len(train_y))
    # print(d1)
    # print(d2)
for i in range(len(psizes)):
    print('in the pneumonia for loop')
    print(psizes[i])
    # print(train_x[0][0])
    # print(sizes[0])
    # print(len(train_x[0]))
    max_rows = 2916
    max_columns = 2663
    masked_value = -1
    # v actually supposed to be len(train_x)
for i in range(len(ptx)):
    ptx[i] = np.ndarray.tolist(ptx[i])
    arr = [-1] * 2663
    arr = np.array(arr)
    print(sizes[i])
    print(i)
    # print(sizes[i])
    for j in range(len(ptx[i])):
        while (len(ptx[i][j]) < 2663):
            ptx[i][j].append(-1)
    while (len(ptx[i]) < 2916):
        ptx[i].append(arr)
    ptx[i] = np.array(ptx[i])
    actual_train.append(ptx[i])
print(len(actual_train))'''
    #print(sizes[678])
#print(sizes[678])


    #input('code break')
    #input('code break')
#max index at 678
#0 is black 255 is white
#22
#1320
ptx = []
pty = []
d1 = 0
d2 = 0
psizes = []
for filename in os.listdir(pneumonia_dir):
    pic = PIL.Image.open('chest_xray/train/PNEUMONIA/'+filename)
    pic_rgb = pic.convert("L")
    size = np.array(pic_rgb.size)
    psizes.append(size)    #print(size)
    if(size[0] > d1):
        d1 = size[0]
    if(size[1] > d2):
        d2 = size[1]
    cache = np.array(pic_rgb)

    #print(cache)
    cache = cache.reshape(size[0],size[1])
    #print(cache.shape)
    #input('code break')
    '''for i in range((size[0])):
        for j in range((size[1])):
            cache.append(np.sum(np.array(pic_rgb.getpixel((i,j))))/3)
            print(np.sum(np.array(pic_rgb.getpixel((i,j))))/3)'''
    ptx.append(cache)
    pty.append(0)
psizes = np.array(psizes)

#print(len(train_x))
#print(len(train_y))
#print(d1)
#print(d2)
'''for i in range(len(sizes)):
    print(sizes[i])'''
#print(train_x[0][0])
#print(sizes[0])
#print(len(train_x[0]))
max_rows = 2916
max_columns = 2663
masked_value = -1
#v actually supposed to be len(train_x)
#actual_train = []
for i in range(2):
    print('in pneumonia sizes')
    print(psizes[i])
    print(i)
    ptx[i] = np.ndarray.tolist(ptx[i])
    arr = [-1] * 2663
    arr = np.array(arr)
    #print(sizes[i])
    #print(i)
    #print(sizes[i])
    for j in range(len(ptx[i])):
        while(len(ptx[i][j]) < 2663):
            ptx[i][j].append(-1)
    while(len(ptx[i])<2916):
        ptx[i].append(arr)
    ptx[i] = np.array(ptx[i])
    actual_train.append(ptx[i])
    train_y.append(1)

print(np.array(actual_train).shape)
print(actual_train)
print(train_y)
print(len(actual_train))
print(len(train_y))
actual_train = np.array(actual_train)
train = []
train_y = np.array(train_y)
for i in range(len(actual_train)):
    print(actual_train[i].shape)
    temp = actual_train[i].reshape(1,7765308)
    train.append(temp)
train = np.array(train)
print(train.shape)
input('code break')
#train_y = tf.squeeze(train_y, axis=-1)


model = Sequential()
model.add(Masking(mask_value = -1, input_shape = (1,7765308)))
model.add(LSTM(972, activation='swish', return_sequences=True, input_shape = (1,7765308)))
model.add(LSTM(3, activation='swish', return_sequences=True))
model.add(Dense(1, activation='relu'))

learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(0.001)
optimizer.learning_rate.assign(learning_rate)

model.compile(optimizer=optimizer, loss='mse')

history = model.fit(train, train_y, epochs=15, verbose=1, batch_size = 1)