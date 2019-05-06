#%% Image processing
import cv2
import glob2

#%%

import math
from tensorflow import keras

import cv2
import numpy as np
from keras import Model
from keras.models import Sequential, load_model
from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input
from keras.layers import MaxPooling2D, Conv2D, Reshape, Dense, Flatten
import tensorflow as tf
#from keras.datasets import cifar10
import sys
import matplotlib.pyplot as plt

import h5py #to save data

#%% Import TrashNet dataset
def display_rgb_image(image):
	plt.imshow(image)
	plt.show()

def load_blur_img(path, size):
	"""
	path - Path to image.
	size - (x, y) tuple.
	"""
	img = cv2.imread(path)

	if img is None:
		print(path)

	img = cv2.blur(img,(5,5))
	img = cv2.resize(img, size)
	return img

def load_img_class(class_paths, class_lable, img_size):
	x = list()
	y = list()

	for path in class_paths:
		img = load_blur_img(path, img_size)

		if img is not None:
			x.append(img)
			y.append(class_lable)

	return np.asarray(x), np.asarray(y)

def load_data(img_size):
	plastic_paths = glob2.glob('./dataset/plastic/*.jpg')
	x_plastic, y_plastic = load_img_class(plastic_paths, 0, img_size)

	return x_plastic, y_plastic


#%%



# Class names for different classes
class_names = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']
trainResized = np.zeros((numTrainImg, minSize, minSize, trainDepth))
testResized = np.zeros((numTestImg, minSize, minSize, testDepth))    

for i in range(len(train_images)):
    trainResized[i] = cv2.resize(train_images[i],dsize=(minSize,minSize),interpolation=cv2.INTER_CUBIC)
for i in range(len(test_images)):
    testResized[i] = cv2.resize(test_images[i],dsize=(minSize,minSize),interpolation=cv2.INTER_CUBIC)
print ('TrainResized number of images:', len(trainResized) , 'TrainResized size X , Y:',len(trainResized[0][1]),',',len(trainResized[0][0]))


#MobileNetV2 download.
#Change input_shape to (maxSize,maxSize,trainDepth) & alpha to 1.4 to get better accuracy!
mobileNetV2 = MobileNetV2(input_shape=(minSize,minSize,trainDepth), alpha=1.0, depth_multiplier=1,
                          include_top=False, weights='imagenet', input_tensor=None, pooling=None, classes=10)

#Using pre trained weights, not looking to tune them.
for layer in mobileNetV2.layers:
    layer.trainable = False

mobileNetV2.summary()


model = Sequential()
model.add(mobileNetV2)

""" Add layers here, dont forget output layer.
model.add(Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='relu'))
model.add(Conv2D(64, kernel_size=(1, 1), strides=(1, 1), activation='relu'))

model.add(Dense(500, activation='relu'))
model.add(Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='relu'))
model.add(Conv2D(64, kernel_size=(1, 1), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='relu'))
model.add(Conv2D(64, kernel_size=(1, 1), strides=(1, 1), activation='relu'))
model.add(Conv2D(128, kernel_size=(1, 1), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))


model.add(Flatten())
model.add(Dense(10000, activation='relu'))
model.add(Dense(10, activation='softmax'))
"""

model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(0.0001), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#Load previous weights
#model.load_weights('Data/model.h5', by_name=True)

#Run the stochastic gradient descent for specified epochs
epochs = 100
batch_size = 64
history = model.fit(trainResized, train_labels, batch_size=batch_size, epochs=epochs) #train & get history?

#Save new weights
#model.save_weights('Data/model.h5')

#Run test
test_loss, test_acc = model.evaluate(testResized, test_labels)

#See the test accuracy
print('Test accuracy:', test_acc)

# Get all predictions for test data
predictions = model.predict(test_images)
predictions.shape


#Get history of loss and accuracy during training and display it with graphs
train_loss = history.history['loss']
train_acc  = history.history['acc']

print('A graph displaying the loss over training epochs')
plt.plot(train_loss)
plt.ylabel('Loss')
plt.xlabel('Epoch number')
plt.show()
plt.savefig('Data/train_loss.png')

print('A graph displaying the accuracy over training epochs')
plt.plot(train_acc)
plt.ylabel('Accuracy')
plt.xlabel('Epoch number')
plt.savefig('Data/train_acc.png')
plt.show()

#Get accuracy for every class during tests

scores = np.zeros(len(class_names))
numPredictions = np.zeros(len(class_names))

for i in range(numTestImg):
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if(predicted_label == true_label):
        scores[true_label] += 1
    numPredictions[true_label] += 1

for i in range(len(class_names)):
    print('Test accuracy for ', class_names[i], 'is :', scores[i]/numPredictions[i])