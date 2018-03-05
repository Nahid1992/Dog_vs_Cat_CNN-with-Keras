import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import numpy as np
import matplotlib.pyplot as plt
import parser

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


batch_size = 16
img_height = 150
img_width = 150

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',  
        target_size=(img_width, img_height),  
        batch_size=batch_size,
        class_mode='binary')  


validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

model = Sequential()

model.add(Conv2D(32,(3,3), input_shape=(img_width,img_height,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

#prevent overfitting - dropout
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#Compile Method
model.compile(loss='binary_crossentropy',
			  optimizer='rmsprop',
			  metrics=['accuracy'])

#Train Model
model.fit_generator(train_generator,
					samples_per_epoch=2048,
					nb_epoch=30,
					validation_data=validation_generator,
					nb_val_samples=832)

model.save_weights('models/simple_CNN.h5')

'''
#Need to load model
#Test Model
img = image.load_img('test1/img001.jpg',target_size=(224,224))
prediction = model.predict(img)
print (prediction)
'''