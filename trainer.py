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
import os
import cv2
import numpy as np
from random import shuffle



class Trainer:
    
    def __init__(self):
        
        self.IMG_SIZE=50
        
        self.categories=['Bread','Burger','Cake','Eggs','Fish','Noodles','Oisters','Pizza','Rice','Rottie','Soup']
        
        self.dataset=[]
        self.data=[]
        self.target=[]
        
        self.target_dict={'Bread':[1,0,0,0,0,0,0,0,0,0,0],
                     'Burger':[0,1,0,0,0,0,0,0,0,0,0],
                     'Cake':[0,0,1,0,0,0,0,0,0,0,0],
                     'Eggs':[0,0,0,1,0,0,0,0,0,0,0],
                     'Fish':[0,0,0,0,1,0,0,0,0,0,0],
                     'Noodles':[0,0,0,0,0,1,0,0,0,0,0],
                     'Oisters':[0,0,0,0,0,0,1,0,0,0,0],
                     'Pizza':[0,0,0,0,0,0,0,1,0,0,0],
                     'Rice':[0,0,0,0,0,0,0,0,1,0,0],
                     'Rottie':[0,0,0,0,0,0,0,0,0,1,0],
                     'Soup':[0,0,0,0,0,0,0,0,0,0,1]}
        

    def create_files(self):
        
        
        for category in self.categories:
        
            path = os.path.join('resized_images',category)
            img_names=os.listdir(path)
            
            for img_name in img_names:
                try:
                    img_path=os.path.join(path,img_name)
                    img=cv2.imread(img_path)
                
                    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                
                    self.dataset.append([img,self.target_dict[category]])
                
                except Exception as e:
                    print(e)
        
        
        shuffle(self.dataset)
        
        for features,label in self.dataset:
            
            self.data.append(features)
            self.target.append(label)
        
        self.data=np.array(self.data)
        self.target=np.array(self.target)
        
        self.data=self.data.reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        
        
        pickle.dump(self.data,open('data1.pickle','wb'))
        pickle.dump(self.target,open('target1.pickle','wb'))
    
    
    def train(self):
        
        
        self.create_files()
        
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
        
        model.add(Flatten())
        
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
        
