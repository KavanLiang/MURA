import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, cohen_kappa_score
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import to_categorical
from keras import regularizers
from keras import optimizers
import numpy as np
import glob
import csv
import pickle
from tqdm import *
import random
import os
import re

from time import time

SEED = 1337

IMAGE_DIM = 200
DATA_TYPE = 'float16'
MODEL_NAME = f'reg{IMAGE_DIM}-{DATA_TYPE}-weights-c64x{3}-c128x{4}-c256x{8}-c256x{8}-c512x{12}-c512x{12}-d{4096}x{2}'

aug = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True)


class Metrics(Callback):

    def __init__(self):
        super().__init__()
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.kappas = []

    def on_train_begin(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict, average='micro')
        _val_recall = recall_score(val_targ, val_predict, average='micro')
        _val_precision = precision_score(val_targ, val_predict, average='micro')
        _val_cohens_kappa = cohen_kappa_score(val_targ.argmax(axis=1), val_predict.argmax(axis=1))
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print(
            f' - val_f1: {_val_f1} - val_precision: {_val_precision} - val_recall: {_val_recall} - val_cohens_kappa: {_val_cohens_kappa}')
        return


callbacks = [
    Metrics(),
    ModelCheckpoint(f'Models/{MODEL_NAME}/' + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
                    save_best_only=True,
                    verbose=1),
    TensorBoard(log_dir=f'logs/{MODEL_NAME}')
]


def mkdir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def model():
    ret = Sequential()

    ret.add(Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', input_shape=(IMAGE_DIM, IMAGE_DIM, 1)))
    ret.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(BatchNormalization())
    ret.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    ret.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(BatchNormalization())

    ret.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(BatchNormalization())

    ret.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
    ret.add(BatchNormalization())

    ret.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(BatchNormalization())

    ret.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(BatchNormalization())

    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(Conv2D(512, kernel_size=(3, 3), padding='same', activation='relu'))
    ret.add(BatchNormalization())

    ret.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    ret.add(Flatten())
    ret.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    ret.add(Dropout(0.5))
    ret.add(Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    ret.add(Dropout(0.5))
    ret.add(Dense(2, activation='softmax'))

    optimizer = optimizers.Adam(amsgrad=True)
    ret.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return ret


def preprocess_image(filename):
    im = load_img(filename, color_mode='grayscale', target_size=(IMAGE_DIM, IMAGE_DIM), interpolation='lanczos')
    ret = img_to_array(im, dtype=f'{DATA_TYPE}')
    return ret


def train(prev=None):
    if prev:
        currModel = keras.models.load_model(prev)
    else:
        currModel = model()

    mkdir(f'Models/{MODEL_NAME}/')

    random.seed(SEED)
    np.random.seed(SEED)

    # with open('MURA-v1.1/train_labeled_studies.csv') as csv_file:
    #     X_train = []
    #     y_train = []
    #     csv_reader = csv.reader(csv_file)
    #     for row in tqdm(csv_reader):
    #         X_append, y_append = build_dataset(row[0], int(row[1]))
    #         X_train += X_append
    #         y_train += y_append
    #
    # with open('MURA-v1.1/valid_labeled_studies.csv') as csv_file:
    #     X_val = []
    #     y_val = []
    #     csv_reader = csv.reader(csv_file)
    #     for row in tqdm(csv_reader):
    #         X_append, y_append = build_dataset(row[0], int(row[1]))
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
    # with open(f'tf-{IMAGE_DIM}-{DATA_TYPE}', 'wb') as file:
    #     pickle.dump((X_train, y_train), file, protocol=4)
    #
    # with open(f'vf-{IMAGE_DIM}-{DATA_TYPE}', 'wb') as file:
    #     pickle.dump((X_val, y_val), file, protocol=4)

    print('Loading Train/Val Sets')

    with open(f'tf-{IMAGE_DIM}-{DATA_TYPE}', 'rb') as file:
        load_train = pickle.load(file)
        X_train, y_train = load_train

    with open(f'vf-{IMAGE_DIM}-{DATA_TYPE}', 'rb') as file:
        load_val = pickle.load(file)
        X_val, y_val = load_val

    X_train = X_train / 255.0

    X_val = X_val / 255.0

    class_weights = [1, 1.6]

    y_train = to_categorical(y_train).astype('uint8')
    y_val = to_categorical(y_val).astype('uint8')

    print('Fitting Model')

    if (prev):
        history = currModel.fit_generator(aug.flow(X_train, y_train, batch_size=64), validation_data=(X_val, y_val),
                                          verbose=1,
                                          epochs=1000,
                                          callbacks=callbacks,
                                          steps_per_epoch=len(X_train) / 100,
                                          class_weight={v: k for v, k in enumerate(class_weights)},
                                          initial_epoch=int(
                                              re.search('[0-9]+', re.search('weights\.[0-9]+-', prev).group(0)).group(
                                                  0)))
    else:
        history = currModel.fit_generator(aug.flow(X_train, y_train, batch_size=64), validation_data=(X_val, y_val),
                                          verbose=1,
                                          epochs=1000,
                                          callbacks=callbacks,
                                          steps_per_epoch=len(X_train) / 100,
                                          class_weight={v: k for v, k in enumerate(class_weights)})

    # history = currModel.fit(x=X_train, y=y_train, batch_size=64, epochs=1000, verbose=1, callbacks=callbacks, validation_data=(X_val, y_val), shuffle=True, class_weight={v: k for v, k in enumerate(class_weights)})


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


if __name__ == '__main__':
    train(
        'Models/reg200-float16-weights-c64x3-c128x4-c256x8-c256x8-c512x12-c512x12-d4096x2/reg200-float16-weights-c64x3-c128x4-c256x8-c256x8-c512x12-c512x12-d4096x2weights.412-0.47.hdf5')
