# -*- coding: utf-8 -*-
"""
Created on Sun May 17 19:09:24 2020

@author: SACHUU
"""

import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

import pickle
from random import shuffle


def create_files():
    IMG_SIZE=50
    
    categories=['Bread','Burger','Cake','Eggs','Fish','Noodles','Oisters','Pizza','Rice','Rottie','Soup']
    
    dataset=[]
    data=[]
    target=[]
    
    target_dict={'Bread':[1,0,0,0,0,0,0,0,0,0,0],
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
    
    for category in categories:
    
        path = os.path.join('resized_images',category)
        img_names=os.listdir(path)
        
        for img_name in img_names:
            try:
                img_path=os.path.join(path,img_name)
                img=cv2.imread(img_path)
            
                img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
                dataset.append([img,target_dict[category]])
            
            except Exception as e:
                print(e)
    
    
    shuffle(dataset)
    
    for features,label in dataset:
        
        data.append(features)
        target.append(label)
    
    data=np.array(data)
    target=np.array(target)
    
    data=data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    
    
    pickle.dump(data,open('data.pickle','wb'))
    pickle.dump(target,open('target.pickle','wb'))