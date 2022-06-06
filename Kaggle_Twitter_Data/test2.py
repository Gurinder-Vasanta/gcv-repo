import numpy as np
import pandas as pd
from sklearn import preprocessing as pre
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import normalize
#text = ['happy','sad','meh','meh','great','good','depressed','ehh','mid']
#max length is 359
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': Activation(swish)})
total_sample_numbers = 100000
data = pd.read_csv('data.csv')
categories = []
encoded = []
negatives = np.random.choice(800000,int(total_sample_numbers/2))
positives = random.sample(range(800000,1600000),int(total_sample_numbers/2))
indices = []
indices.append(negatives)
indices.append(positives)
indices = np.array(indices).reshape(total_sample_numbers)
targets = []
#3,6
temp = [2]*(int(total_sample_numbers/2))
t2 = [0]*(int(total_sample_numbers/2))
targets.append(temp)
targets.append(t2)
targets = np.array(targets).reshape(total_sample_numbers)
for i in range(len(indices)):
    cache = data['text'][indices[i]].split(' ')
    #print(cache)
    categories.append(cache)
#print(len(categories))
#categories = "Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D".split(' ')
for i in range(len(categories)):
    #print(len(categories[i][0]))
    #print(len(categories[i]))
    #input('||')
    encoded.append((pre.LabelBinarizer().fit_transform(categories[i])))
for i in range(len(encoded)):
    encoded[i] = (np.pad(encoded[i], ((0, 359 - len(encoded[i])), (0, 359 - len(encoded[i][0]))))).reshape(1,359*359)
encoded = np.array(encoded)
targets = np.array(targets).reshape(total_sample_numbers,1,1)
print(encoded.shape)
print(targets.shape)
'''a = np.array([[1,2,3,4,5],[3,4,5,6,7]])
print(np.pad(a,((0,0),(0,3))))'''
'''y = np.pad(y,((0,0),(0,359-len(y[0]))))
print(np.pad(y,((0,0),(0,359-len(y[0])))))
print(len(y[0]))'''
print(encoded[0])
print(encoded[99999])
model = Sequential()
#model.add(Dense(718, activation='tanh', input_shape = (1,359*359)))
model.add(Dense(359, activation='swish',input_shape=(1,359*359)))
model.add(Dense(1, activation='relu')) #relu
learning_rate = 0.00005 #0.00005
optimizer = tf.keras.optimizers.Adam(0.001)
optimizer.learning_rate.assign(learning_rate)
model.compile(optimizer=optimizer, loss='mse')
history = model.fit(encoded, targets, epochs=1, verbose=1,batch_size=125)
print(model.predict(encoded[0].reshape(1,1,359*359)))
print(model.predict(encoded[99999].reshape(1,1,359*359)))
'''for i in range(40000,60001):
    print(model.predict(encoded[i].reshape(1,1,359*359)))
    if(i==49999): `
        input('stop')'''
#print(y)
#print(temp)

#local 3 is lr of 0.0005 and 3 epochs
#local 5 is lr of 0.0001 and 30 epochs