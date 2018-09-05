import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, MaxPool2D,Activation
import sklearn
import cv2
import argparse
import os
import matplotlib.pyplot as plt
import keras
import tensorflow as tf


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid

def load_image(data_dir, image_file):
    """
    Load RGB images from a file, convert to grayscale and crop the image (removing the sky at the top and the car front at the bottom)
    """
    path = data_dir+ '/' + 'IMG' + '/' + image_file.split('/')[-1]
    image = cv2.imread(path)
    image = image[50:-20, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(image, (320, 90), cv2.INTER_AREA)

def generator(data_dir, image_paths, steering_angles, correction, batch_size=32):
    '''
    generate train data batch and validation data batch
    '''
    num_samples = len(steering_angles)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(image_paths, steering_angles)
        for offset in range(0, num_samples, batch_size):
            image_paths_batch = image_paths[offset:offset+batch_size]
            steering_angles_batch = steering_angles[offset:offset+batch_size]

            images = []
            angles = []

            for image_path, angle in zip(image_paths_batch,steering_angles_batch):
                img_center = load_image(data_dir, image_path[0])
                img_left = load_image(data_dir, image_path[1])
                img_right = load_image(data_dir, image_path[2])
                images.extend((img_center,img_left,img_right))
                steering_center = float(angle)
                # create adjusted steering measurements for the side camera images
                steering_left = steering_center + correction
                steering_right = steering_center - correction
                angles.extend((steering_center, steering_left, steering_right))

                img_center_flipped = np.fliplr(img_center)
                img_left_flipped = np.fliplr(img_left)
                img_right_flipped = np.fliplr(img_right)

                images.extend((img_center_flipped,img_left_flipped,img_right_flipped))

                steering_center_flipped = -steering_center
                steering_left_flipped = -steering_left
                steering_right_flipped = -steering_right
                angles.extend((steering_center_flipped, steering_left_flipped, steering_right_flipped))

            # trim image to only see section with road
            X_train = np.array(images).reshape((-1,90,320,1))
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def build_model(args):

    config = tf.ConfigProto(device_count={"CPU": 6})
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))
    model = Sequential()
    ## Modified NVIDIA model
    # model.add(Lambda(lambda x: x/127.5-1.0, input_shape=(90,320,1)))
    # model.add(Conv2D(24, (5, 5), activation='elu', subsample=(2, 2)))
    # model.add(Conv2D(36, (5, 5), activation='elu', subsample=(2, 2)))
    # model.add(Conv2D(48, (5, 5), activation='elu', subsample=(2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='elu'))
    # model.add(Conv2D(64, (3, 3), activation='elu'))
    # model.add(Dropout(args.keep_prob))
    # model.add(Flatten())
    # model.add(Dense(100, activation='elu'))
    # model.add(Dense(50, activation='elu'))
    # model.add(Dense(10, activation='elu'))
    # model.add(Dense(1))

    # self made
    model.add(Lambda(lambda x: (x / 255.0) - 0.5,input_shape=(90,320,1)))
    # model.add(Cropping2D(cropping=((50,20), (0,0))))
    model.add(Conv2D(filters=3,kernel_size=(5,5)))
    model.add(MaxPool2D(pool_size=(4,4)))
    model.add(Activation('elu'))
    # second conv layer
    model.add(Conv2D(6,(3,3)))
    model.add(MaxPool2D(pool_size=(4,4)))
    model.add(Activation('elu'))
    # third conv layer
    model.add(Conv2D(12,(3,3)))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())

    model.add(Dense(64,activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Dense(32,activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Dense(16,activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Dense(1))

    # # Lenet
    # model.add(Lambda(lambda x: (x / 255.0) - 0.5,input_shape=(90,320,1)))
    # model.add(Conv2D(6, kernel_size=(5, 5),
    #                  activation='elu'))
    # model.add(MaxPooling2D(pool_size=(4, 4)))
    # model.add(Conv2D(12, (5, 5), activation='elu'))
    # model.add(MaxPooling2D(pool_size=(4, 4)))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(256, activation='elu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(128, activation='elu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))
    steps_per_epoch = len(X_train)//args.batch_size
    validation_steps = len(X_valid)//args.batch_size
    print ('steps',len(X_train),len(X_valid),steps_per_epoch,validation_steps)
    history_object = model.fit_generator(generator(args.data_dir, X_train, y_train, args.correction ,args.batch_size),
                        steps_per_epoch=steps_per_epoch,
                        validation_data=generator(args.data_dir, X_valid, y_valid, args.correction, args.batch_size),
                        validation_steps=validation_steps,
                        callbacks=[checkpoint],
                        epochs=args.nb_epoch,
                        verbose=1,max_q_size=1)

    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.ylim([0,0.05])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=5)
    parser.add_argument('-c', help='steering correction',   dest='correction',        type=float, default=0.2)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=32)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-3)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()

