import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
#target values meanings:
#0: negative tweet
#2: neutral tweet
#4: positive tweet
data = pd.read_csv('data.csv')
text = np.array(data['text'])
target = np.array(data['target'])
print(len(target))
#input('ast')
print(text[0])
loop_cap = 25000
words = []
all_words = []
for i in range(loop_cap):
    tweet = text[i].split(' ')
    temp = []
    for j in range(len(tweet)):
        if('@' in tweet[j]):
            tweet[j] = ''
        if('.com' in tweet[j]):
            tweet[j] = ''
        if(';' in tweet[j]):
            tweet[j] = ''
        if('-' in tweet[j]):
            tweet[j] = ''
        if('http://' in tweet[j]):
            tweet[j] = ''
        if('www.' in tweet[j]):
            tweet[j] = ''
    for k in range(len(tweet)):
        if(tweet[k] == ''):
            continue
        else:
            temp.append(tweet[k])
        if(tweet[k] not in all_words):
            all_words.append(tweet[k])
    words.append(temp)
    print(str(i) + ':         ' + str(temp))

print(words)
print(all_words)
print(len(all_words))
#array = ['Neutral', 'Good','Bad']
y = LabelBinarizer().fit_transform(all_words)
print(y)
print(np.ndarray.tolist(y[0]))
indices = []
for i in range(len(y)):
    indices.append(np.ndarray.tolist(y[i]).index(1))
    print(np.ndarray.tolist(y[i]).index(1))

print(indices)
print(len(indices))
print(len(all_words))

'''data = pd.read_csv('data.csv')
dictionary = {0:'Bad',2:'Neutral',4:'Good',0:'Trash',2:'Meh',4:'Nice'}
array = [['Bad',1],['Neutral',2],['Good',3],['Neutral',5],['Bad',6],['Good',10],['Good',14]]
keys = list(dictionary.keys())
values = list(dictionary.values())
b = []
for i in range(len(dictionary)):
    temp = []
    temp.append(values[i])
    temp.append(keys[i])
    b.append(temp)


print(array)
input('stop')
y = OneHotEncoder().fit(array)
print(y.categories_)
s = y.transform([['Bad',3],['Good',2],['Neutral',1]]).toarray()
#s = y.transform([2,'Hi'],[3,'Goodbye'])
#z = OneHotEncoder().inverse_transform(y[0]).toarray()
print(s)'''