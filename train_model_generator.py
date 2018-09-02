import os
import csv
import matplotlib.pyplot as plt
import keras
import tensorflow as tf

samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:]

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    idx = 0
    print('generator initiated')
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # print('generator yielded a batch %d' % idx)
            for batch_sample in batch_samples:
                img_center = cv2.imread('data/IMG/' + batch_sample[0].split('/')[-1])
                img_left = cv2.imread('data/IMG/' + batch_sample[1].split('/')[-1])
                img_right = cv2.imread('data/IMG/' + batch_sample[2].split('/')[-1])
                img_center_gray = cv2.cvtColor(img_center, cv2.COLOR_BGR2GRAY)
                img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
                img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
                images.extend((img_center_gray,img_left_gray,img_right_gray))
                steering_center = float(line[3])
                # create adjusted steering measurements for the side camera images
                correction = 0.2 # this is a parameter to tune
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                angles.extend((steering_center, steering_left, steering_right))

                img_center_flipped = np.fliplr(img_center)
                img_left_flipped = np.fliplr(img_left)
                img_right_flipped = np.fliplr(img_right)

                img_center_flipped_gray = cv2.cvtColor(img_center_flipped, cv2.COLOR_BGR2GRAY)
                img_left_flipped_gray = cv2.cvtColor(img_left_flipped, cv2.COLOR_BGR2GRAY)
                img_right_flipped_gray = cv2.cvtColor(img_right_flipped, cv2.COLOR_BGR2GRAY)
                images.extend((img_center_flipped_gray,img_left_flipped_gray,img_right_flipped_gray))

                steering_center_flipped = -steering_center
                steering_left_flipped = -steering_left
                steering_right_flipped = -steering_right
                angles.extend((steering_center_flipped, steering_left_flipped, steering_right_flipped))

            # trim image to only see section with road
            X_train = np.array(images).reshape((-1,160,320,1))
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            # idx += 1

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Activation, Lambda, Cropping2D

config = tf.ConfigProto(device_count={"CPU": 6})
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5,input_shape=(160,320,1)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(filters=3,kernel_size=(5,5)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Activation('relu'))
# second conv layer
model.add(Conv2D(6,(4,3)))
model.add(MaxPool2D(pool_size=(4,4)))
model.add(Activation('relu'))
# third conv layer
# model.add(Conv2D(128,(6,6)))
# model.add(MaxPool2D(pool_size=(4,4)))
# model.add(Activation('relu'))
# first fully connnected layer
model.add(Flatten())

model.add(Dense(2340,activation='relu'))
# model.add(Dense(629,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

from keras.callbacks import ModelCheckpoint
filepath="weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
# model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)
print(len(train_samples),int(6*len(train_samples)/32))
history_object = model.fit_generator(train_generator, steps_per_epoch=int(6*len(train_samples)/32), validation_data=validation_generator, validation_steps=int(6*len(validation_samples)/32), callbacks=[checkpoint],epochs=10,use_multiprocessing=True,workers=6)
### print the keys contained in the history object
print(history_object.history.keys())
### plot the training and validation loss for each epoch
# model.load_weights(filepath)
model.save('model.h5')
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.ylim([0,0.1])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# Epoch 1/4
# 10200/10200 [==============================] - 1342s 132ms/step - loss: 0.0048 - val_loss: 7.1966e-04
# Epoch 2/4
# 10200/10200 [==============================] - 1331s 130ms/step - loss: 1.5918e-04 - val_loss: 4.9002e-04
# Epoch 3/4
# 10200/10200 [==============================] - 1339s 131ms/step - loss: 3.9168e-05 - val_loss: 3.0811e-04
# Epoch 4/4
# 10200/10200 [==============================] - 1346s 132ms/step - loss: 2.0874e-05 - val_loss: 2.8142e-04
