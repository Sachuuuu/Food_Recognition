# -*- coding: utf-8 -*-
"""
Created on Sun May 17 19:14:02 2020

@author: SACHUU
"""

import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from matplotlib import pyplot as plt


def train():
    data=pickle.load(open('data.pickle','rb'))
    target=pickle.load(open('target.pickle','rb'))
    
    print(target[0])
    
    print(data.shape)
    
    data=data/255.0
    
    model = Sequential()
    
    model.add(Conv2D(256, (3, 3), input_shape=data.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    
    model.add(Dense(64))
    
    model.add(Dense(11))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    train_history=model.fit(data, target, epochs=30, validation_split=0.3)
    
    model.save_weights('Food_V1.h5')
    
    
    plt.plot(train_history.history['loss'])
    plt.show()