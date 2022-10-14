import PIL.Image
import os
import numpy as np
import tensorflow as tf
normal_dir = 'chest_xray/train/NORMAL'
pneumonia_dir = 'chest_xray/train/PNEUMONIA'
train_x = []
train_y = []
d1 = 0
d2 = 0
sizes = []
for filename in os.listdir(pneumonia_dir):
    pic = PIL.Image.open('chest_xray/train/PNEUMONIA/'+filename)
    pic_rgb = pic.convert("L")
    size = np.array(pic_rgb.size)
    sizes.append(size)    #print(size)
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
    train_x.append(cache)
    train_y.append(0)
sizes = np.array(sizes)
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
for i in range(len(train_x)):
    print(sizes[i])
    print(i)
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
#print(actual_train)
print(actual_train[0])