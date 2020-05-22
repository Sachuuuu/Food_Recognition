# -*- coding: utf-8 -*-
"""
Created on Sun May 17 19:17:50 2020

@author: SACHUU
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import cv2

def creat_model():
    model = Sequential()
    
    model.add(Conv2D(256, (3, 3), input_shape=(50,50,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(64))
    
    model.add(Dense(11))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

def run(path):
    model = creat_model()
    model.load_weights('Food_V1.h5')
    target_dict={0:'Bread',1:'Burger',2:'Cake',3:'Eggs',4:'Fish',5:'Noodles',6:'Oisters',7:'Pizza',8:'Rice',9:'Rottie',10:'Soup'}
    

    img_path = path
    img=cv2.imread(img_path)
    test_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    (height,width)=img.shape[:2]
    test_img=cv2.resize(test_img,(50,50))
    test_img=test_img/255.0
    test_img=test_img.reshape(-1, 50, 50, 1)
    result=model.predict([test_img])
    result=np.argmax(result)
    return(target_dict[result])