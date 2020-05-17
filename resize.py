import os
import cv2
import numpy as np  

categories=['Bread','Burger','Cake','Eggs','Fish','Noodles','Oisters','Pizza','Rice','Rottie','Soup']


for category in categories:
    
    food_path = os.path.join(os.getcwd(),"Foods")

    path = os.path.join(food_path,category)
    img_names=os.listdir(path)
    
    resized_images = os.path.join(os.getcwd(),"resized_images")
    
    for img_name in img_names:

        img_path=os.path.join(path,img_name)
        img=cv2.imread(img_path,0)

        try:
        
            img=cv2.resize(img,(50,50))

            save_path=os.path.join(resized_images,category,img_name)
            #print(save_path)
            cv2.imwrite(save_path,img)

        
        except:

            pass
