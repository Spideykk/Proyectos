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
from keras.layers import Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import random
import pickle
import os

np.random.seed(0)
with open('/home/spideykk/Documentos/Proyectos/SelfDriving/german-traffic-signs/train.p','rb') as f:
    train_data = pickle.load(f)
with open('/home/spideykk/Documentos/Proyectos/SelfDriving/german-traffic-signs/valid.p','rb') as f:
    val_data = pickle.load(f)
with open('/home/spideykk/Documentos/Proyectos/SelfDriving/german-traffic-signs/test.p','rb') as f:
    test_data = pickle.load(f)
    