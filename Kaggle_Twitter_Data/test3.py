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
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import normalize
#text = ['happy','sad','meh','meh','great','good','depressed','ehh','mid']
#max length is 359
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
get_custom_objects().update({'swish': Activation(swish)})
total_sample_numbers = 175000
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
all_tweets = []
#temp is negatives; t2 is positives;
temp = [0]*(int(total_sample_numbers/2))
t2 = [1]*(int(total_sample_numbers/2))
targets.append(temp)
targets.append(t2)
targets = np.array(targets).reshape(total_sample_numbers)
for i in range(len(data)):
    cache = data['text'][i].split(' ')
    #print(cache)
    for j in range(len(cache)):
        categories.append(cache[j])
    print(i)
print(len(categories))
categories = np.unique(np.array(categories))
print(len(categories))
raw_tweets = []
for i in range(len(indices)):
    raw_tweets.append((data['text'][indices[i]]))
#print(raw_tweets)
#print(np.array(raw_tweets).shape)
for i in range(len(raw_tweets)):
    raw_tweets[i] = raw_tweets[i].split(' ')
#print(raw_tweets)
for i in range(len(raw_tweets)):
    #print(len(categories[i][0]))
    #print(len(categories[i]))
    #input('||')
    encoded.append((pre.LabelBinarizer().fit_transform(raw_tweets[i])))
for i in range(len(encoded)):
    encoded[i] = (np.pad(encoded[i], ((0, 110 - len(encoded[i])), (0, 110 - len(encoded[i][0]))))).reshape(1,110*110)
encoded = np.array(encoded)
print(encoded.shape)
targets = targets.reshape(total_sample_numbers,1,1)
model = Sequential()
model.add(Dense(110*10, activation='tanh',input_shape=(1,110*110)))
#model.add(Dense(110*5, activation='PReLU'))
model.add(Dense(110*2, activation='tanh'))
model.add(Dense(1, activation='tanh')) #relu
learning_rate = 0.0005
optimizer = tf.keras.optimizers.Adam(0.001)
optimizer.learning_rate.assign(learning_rate)
model.compile(optimizer=optimizer, loss='mae', metrics=['mse','acc'])
history = model.fit(encoded, targets, epochs=20, verbose=1)
results = open('results10to2epochs20samples175ktanh.csv','w')
results.write('Original Tweet Index,Actual Value,Predicted Value\n')
accuracy_counter = 0
#5 to 1: train acc is .8288
for i in range(len(encoded)):
    results.write(str(indices[i])+','+str(targets[i][0][0])+','+str(model.predict(encoded[i].reshape(1,1,110*110))[0][0][0])+'\n')
    if(int(targets[i][0][0]) == round(model.predict(encoded[i].reshape(1,1,110*110))[0][0][0])):
        accuracy_counter += 1
    print(str(targets[i][0][0])+'<------ actual        predicted ------->' +str(model.predict(encoded[i].reshape(1,1,110*110))[0][0][0]))
results.write('Accuracy: ' + str(accuracy_counter)+str('/')+str(total_sample_numbers) + str(' = ')+str(accuracy_counter/total_sample_numbers))




'''train_x = np.array(encoded).reshape(np.array(encoded).shape[0],1,np.array(encoded).shape[1])
print(train_x)
print(train_x.shape)
print(targets.shape)'''