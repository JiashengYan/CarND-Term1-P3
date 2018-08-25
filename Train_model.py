import csv
import cv2
import numpy as np
import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines[1:]:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    image_flipped = np.fliplr(image)
    images.append(image_flipped)
    measurement = float(line[3])
    measurement_flipped = -measurement
    measurements.append(measurement)
    measurements.append(measurement_flipped)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Activation, Lambda, Cropping2D

### LeNet
model = Sequential()
# first conv layer input=(160,320,3) output=
model.add(Lambda(lambda x: (x / 255.0) - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(filters=12,kernel_size=(5,5)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Activation('relu'))
# second conv layer
model.add(Conv2D(32,(4,3)))
model.add(MaxPool2D(pool_size=(4,4)))
model.add(Activation('relu'))
# third conv layer
# model.add(Conv2D(128,(6,6)))
# model.add(MaxPool2D(pool_size=(4,4)))
# model.add(Activation('relu'))
# first fully connnected layer
model.add(Flatten())

model.add(Dense(12480,activation='relu'))
# model.add(Dense(629,activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2 )
model.save('model.h5')
