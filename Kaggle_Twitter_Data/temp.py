import pandas as pd
import numpy as np
data = pd.read_csv('data.csv')
max = 0
for i in range(len(data)):
    temp = data['text'][i]
    if(len(temp)>max):
        max = len(temp)
print(max)