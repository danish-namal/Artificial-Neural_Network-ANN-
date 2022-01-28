#Code MNN: by Danish Khan
#Date: 12/12/2021

import sys
from typing import Counter
import numpy
from sklearn.neural_network import MLPClassifier
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as image
#feature4 is DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
import os
numpy.set_printoptions(threshold=sys.maxsize)
#Train data directory where data folder kept 
Train_Diractory ="D:\\University files\\Semester-5\\Artificial Intelligence\\Classification_NeuralNetwork\\data\\data\\train"
trainFolder= os.listdir(Train_Diractory)
# print(trainFolder) test

image_List = []
dark_part_count = []
count_images = []
for folders in trainFolder:
    # print(folders) test
    for i in os.listdir(Train_Diractory +"\\"+folders):
        if any([i.endswith(x) for x in ['.png']]):
#image read of all folders
            img = np.array(image.imread(Train_Diractory+"\\"+folders+"\\"+i))
            # image_List.append(img)
#feature1 = crop image small rectangle and pass it to MLP
            crop = img[100:200, 150:300]
            image_List.append(crop)
            
#Count darker part of all images which is 0 or x_train, i.e in first 1 image contaion 1181 darker part.
#feature2 = for whole image shaded    
for i in image_List:
    # n_zeros = np.count_nonzero(np.round(i)==0)
    # dark_part_count.append(n_zeros)
 
 #feature3 = for diagonal shaded part   
    diag=np.diagonal(i)
    diag_zeros = np.count_nonzero(np.round(diag)==0)
    dark_part_count.append(diag_zeros)
        
    
x_train = dark_part_count
# print(x_train,'xt')

# no of data in each folder i.e in f1 261 1111....... y_train = 1,1,1....21, 2.2.2..
for folders in trainFolder:
        c_directary = os.listdir(Train_Diractory+"\\"+folders)
        count_images.append(len(c_directary))
# print(count_images,'iamges') 
val = 1
y_train = []
for values in count_images:
    for j in range(values):
        y_train.append(val)
    val = val+1
# print(y_train,'yt')
    
#<<<<<<<<<<<<<----------------------------------------val data----------------------------------------->>>>>>>>>>>>>>>

# #Train  val data directory where data folder kept 
Train_Diractory ="D:\\University files\\Semester-5\\Artificial Intelligence\\Classification_NeuralNetwork\\data\\data\\val"
trainFolder= os.listdir(Train_Diractory)
# print(trainFolder)

image_List = []
dark_part_count = []
count_images = []
for folders in trainFolder:
    # print(folders) test
    for i in os.listdir(Train_Diractory +"\\"+folders):
        if any([i.endswith(x) for x in ['.png']]):
            #image in all forder read
            img = np.array(image.imread(Train_Diractory+"\\"+folders+"\\"+i))
            image_List.append(img)
            
#Count darker part of all images which is 0 or x_train, i.e in first 1 image contaion 1181 darker part
for i in image_List:
    n_zeros = np.count_nonzero(np.round(i)==0)
    dark_part_count.append(n_zeros)
x_test = dark_part_count
# print(x_test,'xt')

# no of data in each folder i.e in f1 261 1111....... y_train = 1,1,1....21, 2.2.2..
for folders in trainFolder:
        c_directary = os.listdir(Train_Diractory+"\\"+folders)
        count_images.append(len(c_directary))
# print(count_images,'iamges') 
val = 1
y_test = []
for values in count_images:
    for j in range(values):
        y_test.append(val)
    val = val+1
# print(y_test,'yt')

x_test = x_train
y_test = y_train

# #Reshaping the x-train and y-train data acc to MLP parametric requirement i.e tabular form
x_train = np.array(x_train)
x_train = x_train.reshape(len(x_train), -1)
y_train = np.array(y_train)
y_train = y_train.reshape(len(y_train), -1)

x_test= np.array(x_test)
x_test= x_test.reshape(len(x_test), -1) 
y_test= np.array(y_test)
y_test = y_test.reshape(len(y_test), -1)

classifierr = MLPClassifier(hidden_layer_sizes = (100,40,10), activation = 'logistic',solver='sgd',verbose=True, learning_rate_init = 0.1,max_iter=100).fit(x_train, y_train)
#classifierr=DecisionTreeClassifier().fit(x_train, y_train)
res = classifierr.predict(x_test)
res = np.array(res)
print(res)
accu = 0
for i in range(len(res)):
    accu += 1 if res[i] == y_test[i] else 0
print((accu/len(res)))
