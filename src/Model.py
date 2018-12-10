import keras
from keras.layers import Dense, Flatten, BatchNormalization, Dropout, GlobalAveragePooling2D, ReLU, Input, add, Softmax, \
    LeakyReLU, multiply, Reshape, Activation
from keras.models import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, LearningRateScheduler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, cohen_kappa_score
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import to_categorical
from keras import optimizers
import tensorflow as tf
import numpy as np
import glob
import csv
import pickle
from tqdm import *
import random
import os
import re

from shutil import copyfile

from time import time

SEED = 1337

IMAGE_DIM = 200
DATA_TYPE = 'float16'

NUM_64_BLOCK = 1
NUM_128_BLOCK = 3
NUM_256_1_BLOCK = 5
NUM_256_2_BLOCK = 5
NUM_512_1_BLOCK = 7
NUM_512_2_BLOCK = 7

BATCH_SIZE = 32

NUM_TRAINING_EXAMPLES = 36808
NUM_VALIDATION_EXAMPLES = 3197
CLASS_WEIGHTS = [1, 1.6]

INIT_LEARNING_RATE = 0.001
init_epoch = 0

MODEL_NAME = f'AMSGrad-SGD-SEN{IMAGE_DIM}-{DATA_TYPE}-weights-c64xr{NUM_64_BLOCK}-c128xr{NUM_128_BLOCK}-c256xr{NUM_256_1_BLOCK}-c256xr{NUM_256_2_BLOCK}-c512xr{NUM_512_1_BLOCK}-c512xr{NUM_512_2_BLOCK}-MAXPOOL'

aug = ImageDataGenerator(
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True,
    dtype='float16',
    rescale=1 / 255.0)


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


def step_decay(epoch):
    return INIT_LEARNING_RATE * ((0.1) ** (max(epoch - init_epoch, 0) // 30))


callbacks = [
    ModelCheckpoint(f'Models/{MODEL_NAME}/' + 'weights.{epoch:02d}-{val_loss:.2f}-new_best.hdf5', monitor='val_loss',
                    save_best_only=True,
                    verbose=1),
    ModelCheckpoint(f'Models/{MODEL_NAME}/' + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5, verbose=1),
    LearningRateScheduler(step_decay),
    TensorBoard(log_dir=f'logs/{MODEL_NAME}')
]


def mkdir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def se_block(x, num_filters, ratio=16):
    ret = GlobalAveragePooling2D()(x)
    ret = Reshape((1, 1, num_filters))(ret)
    ret = Dense(num_filters // ratio)(ret)
    ret = LeakyReLU()(ret)
    ret = Dense(num_filters)(ret)
    ret = Activation('sigmoid')(ret)
    return multiply([x, ret])


def se_res_block(x, num_filters, num_convs, kernel_size, strides=(1, 1)):
    res = x
    if strides != (1, 1) or res._keras_shape[-1] != num_filters:
        res = Conv2D(num_filters, (1, 1), padding='same', strides=strides)(x)
    ret = nested_conv_layer(x, num_filters, num_convs, kernel_size, strides=strides)
    ret = se_block(ret, num_filters)
    ret = add([res, ret])
    ret = BatchNormalization()(ret)
    ret = LeakyReLU()(ret)
    return ret


def nested_conv_layer(x, num_filters, num_convs, kernel_size, strides=(1, 1)):
    ret = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    for i in range(num_convs - 1):
        ret = BatchNormalization()(ret)
        ret = LeakyReLU()(ret)
        ret = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same')(ret)
    return ret


def model():
    inp = Input(shape=(IMAGE_DIM, IMAGE_DIM, 1))
    x = inp

    x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(NUM_64_BLOCK - 1):
        x = se_res_block(x, 256, 2, (3, 3))
    x = MaxPooling2D(pool_size=(2, 2))(x)

    for i in range(NUM_128_BLOCK):
        x = se_res_block(x, 128, 2, (3, 3))
    x = MaxPooling2D(pool_size=(2, 2))(x)

    for i in range(NUM_256_1_BLOCK):
        x = se_res_block(x, 256, 2, (3, 3))
    x = MaxPooling2D(pool_size=(2, 2))(x)

    for i in range(NUM_256_2_BLOCK):
        x = se_res_block(x, 256, 2, (3, 3))
    x = MaxPooling2D(pool_size=(2, 2))(x)

    for i in range(NUM_512_1_BLOCK):
        x = se_res_block(x, 512, 2, (3, 3))
    x = MaxPooling2D(pool_size=(2, 2))(x)

    for i in range(NUM_512_2_BLOCK):
        x = se_res_block(x, 512, 2, (3, 3))
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(2)(x)
    x = Softmax()(x)

    ret = Model(inputs=[inp], outputs=x)
    return ret


def train(prev=None):
    global init_epoch
    if prev:
        curr_model = keras.models.load_model(prev)
        init_epoch = int(
            re.search('[0-9]+', re.search('weights\.[0-9]+-', prev).group(0)).group(
                0))
    else:
        curr_model = model()

    optimizer = optimizers.SGD(lr=INIT_LEARNING_RATE)
    curr_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print(curr_model.summary())
    mkdir(f'Models/{MODEL_NAME}/')

    random.seed(SEED)
    np.random.seed(SEED)

    print('Fitting Model')

    if (prev):
        history = curr_model.fit_generator(
            aug.flow_from_directory('trainingdata/', target_size=(IMAGE_DIM, IMAGE_DIM), color_mode='grayscale',
                                    class_mode='categorical', batch_size=32),
            validation_data=aug.flow_from_directory('valdata/', target_size=(IMAGE_DIM, IMAGE_DIM),
                                                    color_mode='grayscale',
                                                    class_mode='categorical', batch_size=32),
            validation_steps=NUM_VALIDATION_EXAMPLES / BATCH_SIZE,
            verbose=1,
            epochs=1000,
            callbacks=callbacks,
            steps_per_epoch=NUM_TRAINING_EXAMPLES / BATCH_SIZE,
            class_weight={v: k for v, k in enumerate(CLASS_WEIGHTS)},
            initial_epoch=init_epoch)
    else:
        history = curr_model.fit_generator(
            aug.flow_from_directory('trainingdata/', target_size=(IMAGE_DIM, IMAGE_DIM), color_mode='grayscale',
                                    class_mode='categorical', batch_size=32),
            validation_data=aug.flow_from_directory('valdata/', target_size=(IMAGE_DIM, IMAGE_DIM), color_mode='grayscale',
                                                    class_mode='categorical', batch_size=32),
            validation_steps=NUM_VALIDATION_EXAMPLES / BATCH_SIZE,
            verbose=1,
            epochs=1000,
            callbacks=callbacks,
            steps_per_epoch=NUM_TRAINING_EXAMPLES / BATCH_SIZE,
            class_weight={v: k for v, k in enumerate(CLASS_WEIGHTS)})


def build_dataset_directories():
    mkdir('trainingdata/')
    mkdir('trainingdata/0/')
    mkdir('trainingdata/1/')
    mkdir('valdata/')
    mkdir('valdata/0/')
    mkdir('valdata/1/')
    with open('MURA-v1.1/train_labeled_studies.csv') as csv_file:
        train_count = 0
        csv_reader = csv.reader(csv_file)
        for row in tqdm(csv_reader):
            train_count = process_and_save_examples(row[0], 'trainingdata/', row[1], train_count)

    with open('MURA-v1.1/valid_labeled_studies.csv') as csv_file:
        val_count = 0
        csv_reader = csv.reader(csv_file)
        for row in tqdm(csv_reader):
            val_count = process_and_save_examples(row[0], 'valdata/', row[1], val_count)


def process_and_save_examples(read_directory_path, write_directory_path, label, count):
    working_count = count
    for file in glob.glob(read_directory_path + '*.png'):
        copyfile(file, f'{write_directory_path}/{label}/{working_count}.png')
        working_count += 1
    return working_count


if __name__ == '__main__':
    train('Models/AMSGrad-SEN200-float16-weights-c64xr1-c128xr3-c256xr5-c256xr5-c512xr7-c512xr7-MAXPOOL/weights.111-0.65-new_best.hdf5')
