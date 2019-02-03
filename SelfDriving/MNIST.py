#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 10:09:02 2019

@author: spideykk
"""

import numpy
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import random

np.random.seed(0)
(X_train, y_train), (X_test,y_test) = mnist.load_data()
num_of_samples = []

def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=784, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))
    model.compile(Adam(lr=0.01),loss = 'categorical_crossentropy',metrics=['accuracy'])
    return model


def leNet_model():
    model = Sequential()
    model.add(Conv2D(30, (5,5), input_shape = (28,28,1), activation ='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(15,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(500,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    model.compile(Adam(lr=0.01),loss='categorical_crossentropy',metrics=['accuracy'])
    return model
    


cols = 5
num_classes = 10
fig, axes = plt.subplots(nrows = num_classes, ncols = cols, figsize=(5,10))
fig.tight_layout()
for i in range(cols):
    for j in range(num_classes):
        x_selected = X_train[y_train == j]
        axes[j][i].imshow(x_selected[random.randint(0,len(x_selected)-1),:,:],cmap=plt.get_cmap('gray'))
y_train= to_categorical(y_train,10)
y_test= to_categorical(y_test,10)
X_train = X_train/255.0
X_test = X_test/255.0
X_train= X_train.reshape(X_train.shape[0],28,28,1)
X_test= X_test.reshape(X_test.shape[0],28,28,1)
model = leNet_model()
print(model.summary())
h = model.fit(X_train,y_train,validation_split =0.1, epochs = 10, batch_size = 400,verbose = 1, shuffle = 'true')