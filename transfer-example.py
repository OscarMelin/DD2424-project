#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:56:55 2019

@author: oscarmelin
"""


# coding: utf-8

#%%

import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


#%%

base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

for layer in base_model.layers:
    layer.trainable = False

x=base_model.output

x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(6,activation='softmax')(x) #final layer with softmax activation


#%%

model=Model(inputs=base_model.input,outputs=preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture


#%%
"""
for layer in model.layers[:20]:
    layer.trainable=False
for layer in model.layers[20:]:
    layer.trainable=True
"""

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
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train = train_generator.n // train_generator.batch_size
step_size_validation = validation_generator.n // train_generator.batch_size

history = model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   validation_data = validation_generator, 
                   validation_steps = step_size_validation,
                   epochs=2)

#Get history of loss and accuracy during training and display it with graphs
train_loss = history.history['loss']
train_acc  = history.history['acc']
val_loss = history.history['val_loss']
val_acc = history.history['val_acc']

print('train_loss:', train_loss)
print('train_acc:', train_acc)
print('val_loss:', val_loss)
print('val_acc:', val_acc)

print('A graph displaying the loss over training epochs')
plt.plot(train_loss)
plt.plot(val_loss)
plt.title('Training loss')
plt.ylabel('Loss')
plt.xlabel('Epoch number')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
plt.savefig('graphs/train_loss.png')

print('A graph displaying the accuracy over training epochs')
plt.plot(train_acc)
plt.plot(val_acc)
plt.title('training accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('graphs/train_acc.png')
plt.show()