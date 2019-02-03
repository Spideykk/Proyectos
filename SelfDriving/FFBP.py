#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 11:36:05 2019

@author: spideykk
"""

import numpy
import matplotlib.pyplot as plt
from sklearn import datasets
from keras.models import Sequential
from keras.models import Dense
from keras.optimizers import Adam


def plot_decision_bondary(X, y, model):
    
    x_span = np.linspace(min(X[:,0]) - 0.25,max(X[:,0]) + 0.25,50)
    y_span = np.linspace(min(X[:,1]) - 0.25,max(X[:,1]) + 0.25,50)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_,yy_ = xx.ravel(),yy.ravel()
    grid = np.c_[xx_,yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx,yy,z)
     

np.random.seed(0)

n_pts = 500
X,y = datasets = datasets.make_circles(n_samples = n_pts, random_state = 123, noise = 0.1, factor = 0.2)
model= Sequential()
model.add(Dense(4, input_shape=(2,), activation ='sigmoid'))
model.add(Dense(1, activation ='sigmoid'))
model.compile(Adam(lr = 0.01), 'binary_crossentropy', metrics =['accuracy'])
h = model.fit(x=X,y=y, verbose = 1, batch_size = 20,  epochs = 100,shuffle = 'true')


plot_decision_bondary(X, y, model)
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])


