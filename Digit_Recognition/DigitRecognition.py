#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 15:55:07 2020

@author: anerker
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')

x,y =mnist['data'],mnist['target']
  
import matplotlib
import matplotlib.pyplot as plt
matplotlib.pyplot.imshow
some_digit = x[3601]
some_digit_image = some_digit.reshape(28,28) #Lets reshape it to plot it
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,interpolation = 'nearest')

## Train and Test the dataset
x_train = x[0:6000] #training the dataset from 0 to 60,000
x_test = x[6000:7000] #testing the dataset from 60,000 till end
y_train = y[0:6000] #training the dataset from 0 to 60,000
y_test = y[6000:7000] #training the dataset from 0 to 60,000

shuffle_index = np.random.permutation(6000)
x_train , y_train = x_train[shuffle_index],y_train[shuffle_index]


## Creating a 2 detector
x_train = x_train.astype(np.int8) #Convert string into interger
y_train = y_train.astype(np.int8) #Convert string into integer

y_train_2 = (y_train==8) # If the value is 8 it will return true otherwise false
y_test_2 = (y_test==8) # If the value is 8 it will return true otherwise false 

clf = LogisticRegression(tol = 0.1,max_iter=10000)
clf.fit(x_train,y_train_2)

clf.predict([some_digit])

from sklearn.model_selection import cross_val_score
a = cross_val_score(clf ,x_train , y_train_2 , cv=3 , scoring="accuracy")




