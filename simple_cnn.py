#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:05:34 2019

@author: oscarmelin
"""
#%%


import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, ELU
from keras.layers.convolutional import Convolution2D
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

#%% Shady macOS hack
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
def deepSemla(inputShape):
    model = Sequential()
    model.add(Convolution2D(16, 8, strides=8, padding='valid', input_shape=inputShape))
    model.add(ELU())
    model.add(Convolution2D(32, 5, strides=5, padding='same'))
    model.add(ELU())
    model.add(Convolution2D(64, 5, strides=5, padding='same'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(6))
    model.add(Activation('softmax'))
    return model

model = deepSemla((224, 224, 3))

#%%

datagen = dict(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
		validation_split=0.13,
        fill_mode='nearest')

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, **datagen)


train_generator = train_datagen.flow_from_directory('./dataset/train',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True,
												 subset='training')

validation_generator = train_datagen.flow_from_directory('./dataset/train',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=train_generator.batch_size,
                                                 class_mode='categorical',
                                                 shuffle=True,
												 subset='validation')

#%%

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train = train_generator.n // train_generator.batch_size
step_size_validation = validation_generator.n // train_generator.batch_size

model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data = validation_generator, 
                   validation_steps = step_size_validation,
                   epochs=5)



















