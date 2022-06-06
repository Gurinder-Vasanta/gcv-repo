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
data = pd.read_csv('data.csv',encoding='latin-1')
targets = np.array(data['target'])
tweets = np.array(data['text'])
all_distinct_words = []
for i in range(50000):
    split_text_arr = tweets[i].split(' ')
    #split_text_arr = split_text_arr[split_text_arr != '']
    for j in range(len(split_text_arr)):
        if(split_text_arr[j] not in all_distinct_words):
            all_distinct_words.append(split_text_arr[j])
    print(i)
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
ats = 0
upsets = 0
angrys = 0
sads = 0
dashes = 0
links = 0
for i in range(len(all_distinct_words)):
    if('@' in all_distinct_words[i]):
        ats += 1
    if('upset' in all_distinct_words[i]):
        upsets +=1
    if('angry' in all_distinct_words[i]):
        angrys += 1
    if('sad' in all_distinct_words[i]):
        sads +=1
    if('-' in all_distinct_words[i]):
        dashes += 1
    if('http://' in all_distinct_words[i] or 'https://' in all_distinct_words[i]):
        links +=1
print(ats)
print('ats above')
print(upsets)
print('upsets above')
print(angrys)
print('angrys above')
print(sads)
print('sads above')
print(dashes)
print('dashes above')
print(links)
print('links above')
'''target_dicts = {}
target_dicts[0] = 0
target_dicts[2] = 0
target_dicts[4] = 0
for i in range(len(targets)):
    target_dicts[targets[i]] += 1
    print(targets[i])
    print(i)
print(target_dicts)'''
#to get the imports to work properly
#virtualenv ENV
#source ENV/bin/activate


#1,350,487 distinct words/characters in all of the 1.6 million messages
#367,462 @s in the messagess (no guarantees but i am 90% sure they are all distinct)
#word "upset" appears 117 times
#word "angry" appears 119 times
#word "sad" appears 1661 times
#dash (-) appears 42499 times
#links (http:// and https://) appear 63841 times
