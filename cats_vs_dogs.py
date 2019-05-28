# -*- coding: utf-8 -*-
"""
Created on Mon May 27 20:22:44 2019

Convolutional neural network processing images of dogs and cats and distinguishing between the two

@author: Radu Chirila
"""
from IPython.display import display
from PIL import Image
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image
import numpy as np
import h5py

#Neural network architecture 
#Convolution -> MaxPooling -> flatten->dense->output
model=Sequential();
model.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation='relu'));
model.add(MaxPooling2D(pool_size=(2,2)));
model.add(Flatten());
model.add(Dense(output_dim=128,activation='relu'));
model.add(Dense(output_dim=1,activation='sigmoid'));
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy']);

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale=1./255);

training_set = train_datagen.flow_from_directory('dataset/training_set',target_size=(64,64),batch_size=32,class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64,64),batch_size=32,class_mode='binary')

model.fit_generator(training_set, steps_per_epoch=8000, epochs=7, validation_data=test_set,validation_steps=800)
#end of training 

#save the model to disk
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")

#testing data with new picture (use in console)
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
