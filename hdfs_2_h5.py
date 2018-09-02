import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Activation, Lambda, Cropping2D




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


filepath = 'weights.01-0.00.hdf5'

model.load_weights(filepath)
model.save('model.h5')
