import numpy as np
import pandas as pd
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
import keras
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from torch.utils.data import Dataset, DataLoader, random_split
import math
import tensorflow as tf

all_data = pd.read_csv('train.csv')
all_test_data = np.array(pd.read_csv('test.csv'))
print(all_test_data[0])
output_file = open('test_results.csv','w')
output_file.write('id,survived,rounded\n')
ofile1 = open('train_results.csv','w')
ofile1.write('id,prediction,rounded_prediction,actual\n')

passengerid = []
test_pclass = []
test_gender = []
test_age = []
test_sibling_spouse = []
test_parents_children = []
test_fare = []
test_embarked = []

pid = []
survived = []
pclass = []
gender = []
age = []
sibling_spouse = []
parents_children = []
fare = []
embarked = []

all_data = np.array(all_data)

for i in range(len(all_data)):
    pid.append(all_data[i][0])
    survived.append(all_data[i][1])
    pclass.append(all_data[i][2])
    gender.append(all_data[i][4])
    age.append(all_data[i][5])
    sibling_spouse.append(all_data[i][6])
    parents_children.append(all_data[i][7])
    fare.append(all_data[i][9])
    embarked.append(all_data[i][11])

for i in range(len(all_test_data)):
    passengerid.append(all_test_data[i][0])
    test_pclass.append(all_test_data[i][1])
    test_gender.append(all_test_data[i][3])
    test_age.append(all_test_data[i][4])
    test_sibling_spouse.append(all_test_data[i][5])
    test_parents_children.append(all_test_data[i][6])
    test_fare.append(all_test_data[i][8])
    test_embarked.append(all_test_data[i][10])
#1 is male; 2 is female
relevant_data = []
relevant_test_data = []
for i in range(len(pclass)):
    pclass[i] = float(pclass[i])

for i in range(len(test_pclass)):
    test_pclass[i] = float(test_pclass[i])

for i in range(len(gender)):
    if(gender[i] == 'male'):
        gender[i] = 1.0
    elif(gender[i] == 'female'):
        gender[i] = 2.0

for i in range(len(test_gender)):
    if(test_gender[i] == 'male'):
        test_gender[i] = 1.0
    elif(test_gender[i] == 'female'):
        test_gender[i] = 2.0

for i in range(len(age)):
    if(math.isnan(age[i])):
        age[i] = 0.0
    else:
        age[i] = float(age[i])

for i in range(len(test_age)):
    if(math.isnan(test_age[i])):
        test_age[i] = 0.0
    else:
        test_age[i] = float(test_age[i])

for i in range(len(sibling_spouse)):
    sibling_spouse[i] = float(sibling_spouse[i])

for i in range(len(test_sibling_spouse)):
    test_sibling_spouse[i] = float(test_sibling_spouse[i])

for i in range(len(parents_children)):
    parents_children[i] = float(parents_children[i])

for i in range(len(test_parents_children)):
    test_parents_children[i] = float(test_parents_children[i])

for i in range(len(fare)):
    fare[i] = float(fare[i])

for i in range(len(test_fare)):
    test_fare[i] = float(test_fare[i])
#1 is C; 2 is Q; 3 is S
for i in range(len(embarked)):
    if(embarked[i] == 'C'):
        embarked[i] = 1.0
    elif(embarked[i] == 'Q'):
        embarked[i] = 2.0
    elif(embarked[i] == 'S'):
        embarked[i] = 3.0
    elif(math.isnan(embarked[i])):
        embarked[i] = 0.0

for i in range(len(test_embarked)):
    if(test_embarked[i] == 'C'):
        test_embarked[i] = 1.0
    elif(test_embarked[i] == 'Q'):
        test_embarked[i] = 2.0
    elif(test_embarked[i] == 'S'):
        test_embarked[i] = 3.0
    elif(math.isnan(test_embarked[i])):
        test_embarked[i] = 0.0

for i in range(len(survived)):
    survived[i] = float(survived[i])

for i in range(len(age)):
    relevant_data.append(pclass[i])
    relevant_data.append(gender[i])
    relevant_data.append(age[i])
    relevant_data.append(sibling_spouse[i])
    relevant_data.append(parents_children[i])
    relevant_data.append(fare[i])
    relevant_data.append(embarked[i])

for i in range(len(test_age)):
    relevant_test_data.append(test_pclass[i])
    relevant_test_data.append(test_gender[i])
    relevant_test_data.append(test_age[i])
    relevant_test_data.append(test_sibling_spouse[i])
    relevant_test_data.append(test_parents_children[i])
    relevant_test_data.append(test_fare[i])
    relevant_test_data.append(test_embarked[i])


#input:
#[pclass-gender-age-sibling_spouse-parents_children-fare-embarked]

#print(pclass)
#print()
#print(gender)
#print()
#print(age)
#print()
#print(sibling_spouse)
#print(parents_children)
#print(embarked)
relevant_data = np.array(relevant_data).reshape(len(age), 1, 7)
relevant_test_data = np.array(relevant_test_data).reshape(len(test_age), 1, 7)
#print(relevant_test_data)
model = Sequential()
model.add(LSTM(891, activation='relu', return_sequences = True, input_shape=(1, 7)))
model.add(LSTM(592, activation='relu', return_sequences=True))
model.add(LSTM(7, activation='relu', return_sequences=True))
model.add(Dense(1))

optimizer = tf.keras.optimizers.Adam(0.001)
optimizer.learning_rate.assign(0.0005)
#0.0005 works best
#1 epoch: 595/891 accuracy
#10 epochs: 690/891 accuracy
#15000 epochs: 871/891 accuracy
model.compile(optimizer=optimizer, loss='mse')
survived = np.array(survived).reshape(len(age),1,1)
history = model.fit(relevant_data, survived, epochs=15000, verbose=1)
#.1024 loss for 891 592 7 and dense 7
train_counter = 0
print(model.predict(relevant_data[0].reshape(1,1,7)))
for i in range(len(relevant_data)):
    print(str(pid[i]) + ',' + str(model.predict(relevant_data[i].reshape(1,1,7))[0][0][0]) + ',' + str(round(model.predict(relevant_data[i].reshape(1,1,7))[0][0][0])) + ','+str(int(survived[i][0][0])))
    if(round(model.predict(relevant_data[i].reshape(1,1,7))[0][0][0]) == int(survived[i][0][0])):
        train_counter += 1
    ofile1.write(str(pid[i]) + ',' + str(model.predict(relevant_data[i].reshape(1,1,7))[0][0][0]) + ',' + str(round(model.predict(relevant_data[i].reshape(1,1,7))[0][0][0])) + ','+str(int(survived[i][0][0])) + '\n')
ofile1.write('Accuracy: ' + str(train_counter) + '/' + str(len(relevant_data)))
for i in range(len(relevant_test_data)):
    print(str(passengerid[i]) + ',' + str(model.predict(relevant_test_data[i].reshape(1,1,7))[0][0][0]) + ',' + str(round(model.predict(relevant_test_data[i].reshape(1,1,7))[0][0][0])))
    output_file.write(str(passengerid[i]) + ',' + str(model.predict(relevant_test_data[i].reshape(1,1,7))[0][0][0]) + ',' + str(round(model.predict(relevant_test_data[i].reshape(1,1,7))[0][0][0])) + '\n')

#for i in range(len(relevant_test_data)):
'''for i in range(len(relevant_data)):
    relevant_data[i][0][0] = int(relevant_data[i][0][0])'''
