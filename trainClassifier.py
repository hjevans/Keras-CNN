# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 09:20:06 2017

@author: Hannah
"""

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plot
from scipy import ndimage
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.models import load_model


with open('trainFeat.pkl','rb') as a:
    feat = pickle.load(a)
with open('trainLabel.pkl','rb') as b:
    label = pickle.load(b)
    
featNormal = (np.float64(feat)-127)/127

model = Sequential()
#model.add... 10
model.add(Convolution2D(20, (5, 5),border_mode = 'same', input_shape = featNormal[0].shape))
model.add(Activation('relu'))
model.add(MaxPooling2D())#(dim_ordering='tf'))

model.add(Convolution2D(20, (5, 5),border_mode = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())#(dim_ordering='tf'))
#model.add... 60
model.add(Convolution2D(40, (5, 5),border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D())#(dim_ordering='tf'))


#repeat Convolution/Activation/Pooling as needed

model.add(Flatten())                 # Flatten features for input to fully connected layers

#model.add(Dense(10))                 # 10 hidden neurons in fully connected layer
#model.add(Activation('relu'))        # Relu activation


model.add(Dense(20))
model.add(Activation('relu'))


model.add(Dense(43))                  # 2 output nodes
model.add(Activation('softmax'))      # Linear activation for regression task

model.compile(Adam(), 'sparse_categorical_crossentropy', ['sparse_categorical_accuracy'])
hist = model.fit(featNormal,label,epochs=1)
while hist.history['sparse_categorical_accuracy'][0] < .99:
    hist = model.fit(featNormal,label,epochs=1)           
pred = model.predict(featNormal)

model.save('model.h5')
'''
cap = cv2.VideoCapture('drive.mp4')
ret,frame = cap.read()
seg = felz(frame,scale=200,min_size=1000)
bound = mark_boundaries(frame,seg)
segBound = mark_boundaries(seg/np.max(seg),seg)
'''
