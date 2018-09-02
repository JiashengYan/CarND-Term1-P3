import csv
import os
import cv2
import numpy as np
import sklearn
from keras.models import Model
import matplotlib.pyplot as plt

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

lines = lines[1:]

from sklearn.model_selection import train_test_split

# lines1, lines2 = train_test_split(lines, test_size=0.2)


# train_samples, validation_samples = train_test_split(lines1, test_size=0.2)

# def generator(samples, batch_size=32):
#     num_samples = len(samples)
#     while 1: # Loop forever so the generator never terminates
#         sklearn.utils.shuffle(samples)
#         for offset in range(0, num_samples, batch_size):
#             batch_samples = samples[offset:offset+batch_size]
#
#             images = []
#             angles = []
#             for batch_sample in batch_samples:
#                 name = 'data/IMG/'+batch_sample[0].split('/')[-1]
#                 center_image = cv2.imread(name)
#                 center_angle = float(batch_sample[3])
#                 images.append(center_image)
#                 angles.append(center_angle)
#
#             # trim image to only see section with road
#             X_train = np.array(images)
#             y_train = np.array(angles)
#             yield sklearn.utils.shuffle(X_train, y_train)
#
# train_generator = generator(train_samples, batch_size=32)
# validation_generator = generator(validation_samples, batch_size=32)



images = []
steering_angles = []
for line in lines[1:]:

    img_center = cv2.imread('data/IMG/' + line[0].split('/')[-1])
    img_left = cv2.imread('data/IMG/' + line[1].split('/')[-1])
    img_right = cv2.imread('data/IMG/' + line[2].split('/')[-1])
    img_center_gray = cv2.cvtColor(img_center, cv2.COLOR_BGR2GRAY)
    img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    images.extend((img_center_gray,img_left_gray,img_right_gray))
    steering_center = float(line[3])
    # create adjusted steering measurements for the side camera images
    correction = 0.2 # this is a parameter to tune
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    steering_angles.extend((steering_center, steering_left, steering_right))

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
    steering_angles.extend((steering_center_flipped, steering_left_flipped, steering_right_flipped))

# import gc
# del lines
# gc.collect()
# from sys import getsizeof
# print(getsizeof(images)/1.0e9)

import psutil
# process = psutil.Process(os.getpid())
# print(process.memory_info().rss)
print(psutil.virtual_memory())
images = np.array(images).reshape((-1,160,320,1))
X_train = images

steering_angles = np.array(steering_angles).reshape((-1,1))
y_train = steering_angles

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Activation, Lambda, Cropping2D, Dropout

### LeNet
model = Sequential()
# first conv layer input=(160,320,3) output=
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
model.add(Dropout(0.5))
# model.add(Dense(629,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

from keras.callbacks import ModelCheckpoint
filepath="weights.{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)

history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10 ,batch_size=32, callbacks=[checkpoint])

# model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=1)
### print the keys contained in the history object
print(history_object.history.keys())
# model.load_weights(filepath)
# model.save('model.h5')
### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.ylim([0,0.1])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


