import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from PIL import Image, ImageOps
import glob
import csv
import pickle
from tqdm import *
import random

import matplotlib.pyplot as plt

SEED = 69

IMAGE_DIM = 200

callbacks = [
    ModelCheckpoint(f'x{IMAGE_DIM}'+'-weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
]

aug = ImageDataGenerator(rotation_range=2, width_shift_range=0.02,
                         height_shift_range=0.02, shear_range=0.02, zoom_range=0.02,
                         horizontal_flip=True)


def model():
    ret = Sequential()

    ret.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', input_shape=(IMAGE_DIM, IMAGE_DIM, 1)))
    ret.add(BatchNormalization())
    ret.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu'))
    ret.add(BatchNormalization())
    ret.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    ret.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    ret.add(BatchNormalization())
    ret.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    ret.add(BatchNormalization())
    ret.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    ret.add(BatchNormalization())
    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    ret.add(BatchNormalization())
    ret.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    ret.add(Flatten())
    ret.add(Dense(1024, activation='relu'))
    ret.add(Dropout(0.3))
    ret.add(Dense(1024, activation='relu'))
    ret.add(Dropout(0.3))
    ret.add(Dense(1, activation='sigmoid'))
    ret.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return ret


def preprocess_image(filename):
    im = load_img(filename, color_mode='grayscale', target_size=(IMAGE_DIM, IMAGE_DIM), interpolation='lanczos')
    ret = img_to_array(im, dtype='float16')
    return ret


def train(prev=None):
    if prev:
        currModel = keras.models.load_model(prev)
    else:
        currModel = model()

    random.seed(SEED)
    np.random.seed(SEED)

    # with open('MURA-v1.1/train_labeled_studies.csv') as csv_file:
    #     X_train = []
    #     y_train = []
    #     csv_reader = csv.reader(csv_file)
    #     for row in tqdm(csv_reader):
    #         X_append, y_append = build_dataset(row[0], row[1])
    #         X_train += X_append
    #         y_train += y_append
    #
    # with open('MURA-v1.1/valid_labeled_studies.csv') as csv_file:
    #     X_val = []
    #     y_val = []
    #     csv_reader = csv.reader(csv_file)
    #     for row in tqdm(csv_reader):
    #         X_append, y_append = build_dataset(row[0], row[1])
    #         X_val += X_append
    #         y_val += y_append
    #
    # X_train = np.array(X_train)
    # y_train = np.array(y_train)
    # X_train, y_train = unison_shuffled_copies(X_train, y_train)
    #
    # X_val = np.array(X_val)
    # y_val = np.array(y_val)
    # X_val, y_val = unison_shuffled_copies(X_val, y_val)
    #
    # with open(f'tf-{IMAGE_DIM}', 'wb') as file:
    #     pickle.dump((X_train, y_train), file, protocol=4)
    #
    # with open(f'vf-{IMAGE_DIM}', 'wb') as file:
    #     pickle.dump((X_val, y_val), file, protocol=4)

    with open(f'tf-{IMAGE_DIM}', 'rb') as file:
        load_train = pickle.load(file)
        X_train, y_train = load_train

    with open(f'vf-{IMAGE_DIM}', 'rb') as file:
        load_val = pickle.load(file)
        X_val, y_val = load_val

    X_train = X_train / 255.0

    X_val = X_val / 255.0

    history = currModel.fit_generator(aug.flow(X_train, y_train, batch_size=64), validation_data=(X_val, y_val),
                                      verbose=1,
                                      epochs=100,
                                      callbacks=callbacks,
                                      steps_per_epoch=len(X_train) / 100)

    # history = currModel.fit(x=X_train, y=y_train, batch_size=64, epochs=100, verbose=1, callbacks=callbacks, validation_data=(X_val, y_val))

    with open('history3', 'wb') as file:
        pickle.dump(history, file)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def build_dataset(directory_path, label):
    X = []
    for file in glob.glob(directory_path + "*.png"):
        X += [preprocess_image(file)]
    y = [label] * len(X)
    return X, y
