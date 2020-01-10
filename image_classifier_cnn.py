# print("hello")
import os
training_folder = "data/shapes/training"
# print(os.getcwd())
classes = sorted(os.listdir(training_folder))
print(classes)

import sys
import keras
print("keras_version", keras.__version__)
from keras import backend as K
img_size = (128, 128)

from keras.preprocessing.image import ImageDataGenerator
batch_size = 30
datagen = ImageDataGenerator(rescale=1.0/255, #normalize the pixel values
                             validation_split=0.3)

train_generator = datagen.flow_from_directory(training_folder,
                                              target_size=img_size,
                                              batch_size=batch_size,
                                              class_mode='categorical',
                                              subset='training') #set as training data

validation_generator = datagen.flow_from_directory(training_folder,
                                                   target_size=img_size,
                                                   batch_size=batch_size,
                                                   class_mode='categorical',
                                                   subset='validation') #validation data set

# define a CNN classifier network
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Flatten, Dense
from keras import optimizers

# the model is defined into a layer of sequence
model = Sequential()
model.add(Conv2D(32, (6, 6), input_shape=train_generator.image_shape, activation='relu' ))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(32, (6, 6), activation='relu' ))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(train_generator.num_classes, activation='softmax'))

# using adam optimizer
opt = optimizers.Adam(lr=0.001)

# with layers defined now  we shallcompile th emodel
model.compile(loss='categorical_crossentropy',
              optimizer = opt,
              metrics=['accuracy'])

print(model.summary)

# train the model over 5 epochs
num_epochs = 5
history = model.fit_generator(train_generator,
                              steps_per_epoch=train_generator.samples,
                              validation_data= validation_generator,
                              validation_steps=validation_generator.samples,
                              epochs=num_epochs)

from keras.models import load_model
model_name = 'shape-classifier.h5'
model.save(model_name)
print("model saved")
del model

