import PIL.Image
import os
import numpy as np
normal_dir = 'chest_xray/train/NORMAL'
pneumonia_dir = 'chest_xray/train/PNEUMONIA'
train_x = []
train_y = []
d1 = 0
d2 = 0
sizes = []
converted = []
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
    converted.append(cache)

print(converted[0][0])
for i in range(len(converted)):
    for j in range(len(converted[i])):

        while(len(converted[i][j])<d2+1):
            converted[i][j] = np.append(converted[i][j],0)
        print(converted[i][j])
    input('FOR LOOP PAUSE')
input('STOP')
#converted = np.array(converted, dtype=object)
'''for i in range(len(converted)):
    for j in range(len(converted[i])):
        for k in range(len(converted[i][j])):
            #print(converted[i][j])
            #input('STOP')
            temp = np.array(converted[i][j])
            print(temp)
            input('temp above')
            while((len(temp))<d2+1):
                #print(len(temp))
                temp = np.append(temp,0)
        
            #print(temp)
        #print(d1)
        #print(d2)
        input('PAUSE')
        #while(len(converted[i][j])
        #print(converted[i][j])
    #input('cb')'''

for i in range(len(converted)):
    zero = [0] * len(converted[i])
    #converted[i] = np.ndarray.tolist(converted[i]).reshape(sizes[i][0],sizes[i][1])
    #converted[i] = np.reshape(sizes[i][0],sizes[i][1])
    print(converted[i].shape)
    print(len(converted[i]))
    converted[i] = np.vstack([converted[i],zero])
    #converted[i].append(zero)
    print(converted[i].shape)
    print(converted[i])
    input('cb')
    while len(converted[i]<d1):
        print(len(converted[i]))
        np.append(converted[i],zero)
    print(len(converted[i]))
    input('cb')
print(converted)
print(converted[0])
print(len(sizes))
print(d1)
print(d2)
#for i in range(len(sizes)):
 #   print(sizes[i].shape)