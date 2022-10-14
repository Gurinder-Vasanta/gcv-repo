import cv2
import os
import PIL.Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': Activation(swish)})

normal_dir = 'chest_xray/train/NORMAL'
pneumonia_dir = 'chest_xray/train/PNEUMONIA'
norm_test = 'chest_xray/test/NORMAL'
pn_test = 'chest_xray/test/PNEUMONIA'
'''for filename in os.listdir(normal_dir):

    pic = PIL.Image.open('chest_xray/train/NORMAL/'+filename)
    pic_rgb = pic.convert("L")
    cv2.resize
    print(np.array(pic_rgb))'''
train_x = []
train_y = []
test_x = []
test_y = []
size = 1000
#def generator(directory, output_arr,size = size)
for filename in os.listdir(normal_dir):
    cv2im = cv2.imread('chest_xray/train/NORMAL/'+filename,0)
    t = (cv2.resize(cv2im,(size,size)))#.reshape(1,500*500)
    #print(np.array(t))
    train_x.append(t)
    #train_y.append([5]*size*size)
    train_y.append(0)
for filename in os.listdir(pneumonia_dir):
    cv2im = cv2.imread('chest_xray/train/PNEUMONIA/'+filename,0)
    t = (cv2.resize(cv2im,(size,size)))#.reshape(1,500*500)
    #print(np.array(t))
    train_x.append(t)
    #train_y.append([10]*size*size)
    train_y.append(1)
for filename in os.listdir(norm_test):
    cv2im = cv2.imread('chest_xray/test/NORMAL/'+filename,0)
    t = (cv2.resize(cv2im,(size,size)))#.reshape(1,500*500)
    #print(np.array(t))
    test_x.append(t)
    #train_y.append([10]*size*size)
    test_y.append(0)
for filename in os.listdir(pn_test):
    cv2im = cv2.imread('chest_xray/test/PNEUMONIA/'+filename,0)
    t = (cv2.resize(cv2im,(size,size)))#.reshape(1,500*500)
    #print(np.array(t))
    test_x.append(t)
    #train_y.append([10]*size*size)
    test_y.append(1)
train_x = np.array(train_x).reshape(5216,1,size*size)
test_x = np.array(test_x).reshape(624,1,size*size)
#train_y = np.array(train_y).reshape(5216,1,size*size)
train_y = np.array(train_y).reshape(5216,1,1)
test_y = np.array(test_y).reshape(624,1,1)
print(train_x.shape)
print(train_y.shape)
#print(np.ndarray.tolist(train_y))
model = Sequential()
model.add((Dense(326, activation='softplus',input_shape=(1,size*size))))
model.add((Dense(163, activation='sigmoid')))
model.add(Dense(1))

learning_rate = 0.005 #0.005
optimizer = tf.keras.optimizers.Adam(0.001)
optimizer.learning_rate.assign(learning_rate)
model.compile(optimizer=optimizer, loss='mse',metrics=['mae','acc'])
history = model.fit(train_x, train_y, epochs=3, verbose=1)
print(model.predict(train_x[0].reshape(1,1,size*size)))
counter_train = 0
for i in range(len(train_x)):
    print(str(i) + ' ' + str(train_y[i][0][0]) + ' ' + str(round(model.predict(train_x[i].reshape(1,1,size*size))[0][0][0])) + ' '  + str(model.predict(train_x[i].reshape(1,1,size*size))[0][0][0]))
    if(round(model.predict(train_x[i].reshape(1,1,size*size))[0][0][0]) == train_y[i][0][0]):
        counter_train += 1
print(str(counter_train) +'/'+'5216 = ' + str(counter_train/5216))
print('--------   train above test below   ---------')
counter_test = 0
for i in range(len(test_x)):
    print(str(i) + ' ' + str(test_y[i][0][0]) + ' ' + str(round(model.predict(test_x[i].reshape(1,1,size*size))[0][0][0])) + ' '  + str(model.predict(test_x[i].reshape(1,1,size*size))[0][0][0]))
    if(round(model.predict(test_x[i].reshape(1,1,size*size))[0][0][0]) == test_y[i]):
        counter_test += 1
print(str(counter_test) +'/'+'623 = ' + str(counter_test/623))