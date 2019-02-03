
"""
Created on Mon Jan 28 11:36:05 2019

@author: spideykk
"""

import numpy
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

def plot_decision_bondary(X, y, model):
    
    x_span = np.linspace(min(X[:,0]) - 1,max(X[:,0]) + 1)
    y_span = np.linspace(min(X[:,1]) - 1,max(X[:,1]) + 1)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_,yy_ = xx.ravel(),yy.ravel()
    grid = np.c_[xx_,yy_]
    pred_func = model.predict_classes(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx,yy,z)

np.random.seed(0)

n_pts = 500
center = [[-1,1],[-1,-1],[1,-1]]
X,y = datasets.make_blobs(n_samples = n_pts, random_state = 123, centers=center, cluster_std = 0.4)
y_cat = to_categorical(y,3)
model = Sequential()
model.add(Dense(units=3, input_shape=(2,),activation='softmax'))
model.compile(Adam(lr = 0.1), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x=X,y=y_cat,verbose = 1, batch_size = 50, epochs = 100)


plot_decision_bondary(X, y, model)
plt.scatter(X[y==0,0],X[y==0,1])
plt.scatter(X[y==1,0],X[y==1,1])
plt.scatter(X[y==2,0],X[y==2,1])