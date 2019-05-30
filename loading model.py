# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:53:19 2019

@author: radus
"""
from IPython.display import display
from PIL import Image
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
import numpy as np
import h5py


with open('model.json','r') as f:
    json = f.read()
model = model_from_json(json)

model.load_weights("model.h5")


def newImage():
    name=input("Name of file: ");
    random_pic=image.load_img(name,target_size=(64,64));
    random_pic=image.img_to_array(random_pic);
    random_pic= np.expand_dims(random_pic,axis=0);
    result=model.predict(random_pic);
    if result[0][0] >= 0.5:
        print("dog");
    else:
        print("cat");
    print(result[0][0])
