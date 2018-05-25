# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 09:39:18 2017

@author: Hannah
"""

import pickle
import numpy as np
import cv2
from skimage.segmentation import felzenszwalb as felz
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
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

with open('trainFeat.pkl','rb') as e:
    feat = pickle.load(e)
with open('trainLabel.pkl','rb') as f:
    label = pickle.load(f)
    
featNormal = (np.float64(feat)-127)/127

figs = 0

model = load_model('model.h5')
modelLogit = Model(input=model.input,outputs=model.get_layer(index=len(model.layers)-1).output)

correct = 0
correctL = 0
acc = [True]*len(label)
accL = [True]*len(label)
pred = model.predict(featNormal)
predL = modelLogit.predict(featNormal)
incorrects = cv2.VideoWriter('incorrect.mp4',cv2.VideoWriter_fourcc(*'XVID'),5,(feat[0].shape[1],feat[0].shape[0]))
for i in range(len(label)):
    if np.where(pred[i]==max(pred[i]))==label[i]:
        # things go here
        correct += 1
    elif np.where(pred[i]==max(pred[i]))!=label[i]:
        # other things go here
        acc[i] = False    
        feat[i] = cv2.cvtColor(feat[i],cv2.COLOR_BGR2RGB)
        # Display incorrect prediction in red on image
        cv2.putText(feat[i],str(np.where(pred[i]==max(pred[i]))[0][0]),(20,10),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,0,255),0)
        # Display ground truth label in green on image
        cv2.putText(feat[i],str(label[i]),(5,10),cv2.FONT_HERSHEY_SIMPLEX,.3,(0,255,0),0)
        cv2.resize(feat[i],(100,100))
        incorrects.write(feat[i])
        
    if np.where(predL[i]==max(predL[i]))==label[i]:
        correctL += 1
    elif np.where(predL[i]==max(predL[i]))!=label[i]:
        accL[i] = False
incorrects.release()
print('Incorrectly classified images generated to incorrect.mp4')
print('Green number = ground truth label')
print('Red number = incorrect prediction')        
accuracy = correct/len(label)
print('Accuracy: ', accuracy)
accuracyL = correctL/len(label)
print('AccuracyL: ' , accuracyL)


ret = True
itrFrame = 0
cap = cv2.VideoCapture('drive.mp4')
ret,frame = cap.read()
rgns = frame.copy()
seg = felz(frame,scale=100,min_size=1000)

out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'XVID'),5,(frame.shape[1],frame.shape[0]))
detect = cv2.VideoWriter('regionProposal.mp4',cv2.VideoWriter_fourcc(*'XVID'),5,(frame.shape[1],frame.shape[0]))
# The following section runs very slowly
while ret==True:
#for q in range(400):
    #cap = cv2.VideoCapture('drive.mp4')
    #ret,frame = cap.read()
    # from project 03 bounding boxes. To modify.
    indices = []
    maybeSignNormal = []
    for m in range(1,np.max(seg)+1):
        ind = np.where(seg==m)
        #xBound = np.max(ind[0]) - np.min(ind[0]) 
        #yBound = np.max(ind[1]) - np.min(ind[1])
        
        # filter out segments based on size
        if np.max(np.where(seg==m)[0]) - np.min(np.where(seg==m)[0]) > 60:
            # was 50
            seg[np.where(seg==m)] = 0
            
        elif np.max(np.where(seg==m)[1]) - np.min(np.where(seg==m)[1]) > 60:
            seg[np.where(seg==m)] = 0
            # was 50
        elif np.max(np.where(seg==m)[0]) - np.min(np.where(seg==m)[0]) < 30:
            seg[np.where(seg==m)] = 0
            
        elif np.max(np.where(seg==m)[1]) - np.min(np.where(seg==m)[1]) < 30:
            seg[np.where(seg==m)] = 0
        
        else:
            maybeSign = frame[np.min(ind[0]):np.max(ind[0]),np.min(ind[1]):np.max(ind[1]),:]
            maybeSign = cv2.cvtColor(maybeSign,cv2.COLOR_BGR2RGB)
            maybeSign = cv2.resize(maybeSign,(32,32))
            maybeSignNormal.append((np.float64(maybeSign)-127)/127)
            indices.append(ind)
            
    if not not maybeSignNormal:
        maybeSignNormal = np.asarray(maybeSignNormal)
        signPred = model.predict(np.array(maybeSignNormal))
        signPredL = modelLogit.predict(np.array(maybeSignNormal))
        print('Softmax frame',itrFrame,':',np.argmax(signPred),np.max(signPred))
        for i in range(len(indices)):
            if np.max(signPredL[i]) > 20:
                # draw bounding box for confident detect on output
                cv2.rectangle(frame,(np.max(indices[i][1]),np.max(indices[i][0])),(np.min(indices[i][1]),np.min(indices[i][0])),
                              (0,255,0),3)
                # draw bounding box for confident detect on region proposal
                cv2.rectangle(rgns,(np.max(indices[i][1]),np.max(indices[i][0])),(np.min(indices[i][1]),np.min(indices[i][0])),
                              (0,255,0),3)
                # write prediction next to box twice to be easily read
                cv2.putText(frame,str(np.argmax(signPred[i])),(np.max(indices[i][1])+5,np.max(indices[i][0])+5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                cv2.putText(frame,str(np.argmax(signPred[i])),(np.min(indices[i][1])-20,np.max(indices[i][0])+5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
            else:
                # draw bounding box for unconfident detect on region proposal
                cv2.rectangle(rgns,(np.max(indices[i][1]),np.max(indices[i][0])),(np.min(indices[i][1]),np.min(indices[i][0])),
                              (0,0,255),3)
    itrFrame += 1
    #frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    
    out.write(frame)
    detect.write(rgns)
    ret,frame = cap.read()
    if ret == True:
        rgns = frame.copy()
        seg = felz(frame,scale=100,min_size=1000)
        # scale was 80
    #bound    = mark_boundaries(frame,seg)                                   
    #segBound = mark_boundaries(seg/np.max(seg),seg)

detect.release()
print('Region proposal bounding boxes generated to regionProposal.mp4')
print('Rgn proposal: green box indicates confident detection')
print('Rgn proposal: red box indicated unconfident detection')
out.release()
print('Final output generated to output.mp4')



