import csv
import glob
import os
import random
import re
from shutil import copyfile

import keras
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.layers import Dense, BatchNormalization, Input, add, ReLU, multiply, Reshape, Activation, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from keras.regularizers import l1_l2
from tqdm import *
from keras.initializers import glorot_normal

SEED = 69
IMAGE_DIM = 200
DATA_TYPE = 'float16'

NUM_64_BLOCK = 3
NUM_128_BLOCK = 7
NUM_256_BLOCK = 28
NUM_512_BLOCK = 7

BOTTLENECK_RATIO = 1

BATCH_SIZE = 32

NUM_TRAINING_EXAMPLES = 36808
NUM_VALIDATION_EXAMPLES = 3197
CLASS_WEIGHTS = [1, 1.6]

INIT_LEARNING_RATE = 1e-4
ALPHA_REG = 1e-5
init_epoch = 0

MODEL_NAME = f'Adam-DECAY-{INIT_LEARNING_RATE}-PREACTIVATION-ImageDIM{IMAGE_DIM}-l1_l2-REG-{ALPHA_REG}-{DATA_TYPE}-weights-c64xr{NUM_64_BLOCK}-c128xr{NUM_128_BLOCK}-' \
             f'c256xr{NUM_256_BLOCK}-c512xr{NUM_512_BLOCK}-MAXPOOL'

aug = ImageDataGenerator(
    rotation_range=45,
    horizontal_flip=True,
    vertical_flip=True,
    dtype='float16'
    )

callbacks = [
    ModelCheckpoint(f'Models/{MODEL_NAME}/' + 'weights.{epoch:02d}-{val_loss:.2f}-new_best.hdf5', monitor='val_loss',
                    save_best_only=True,
                    verbose=0),
    ModelCheckpoint(f'Models/{MODEL_NAME}/' + 'weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=5, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50, verbose=1),
    TensorBoard(log_dir=f'logs/{MODEL_NAME}')
]


def mkdir(file_path):
    """
    Make a new directory

    #Arguments
        file_path: the name of the new directory
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def se_block(x, ratio=16):
    """
    A squeeze excitation implementation

    #Arguments
        x: input layer
        ratio: ratio to scale dense layers
    """
    num_filters = x._keras_shape[-1]
    ret = GlobalAveragePooling2D()(x)
    ret = Reshape((1, 1, num_filters))(ret)
    ret = Dense(num_filters // ratio, kernel_initializer=glorot_normal(SEED), kernel_regularizer=l1_l2(ALPHA_REG, ALPHA_REG),)(ret)
    ret = ReLU()(ret)
    ret = Dense(num_filters, kernel_initializer=glorot_normal(SEED), kernel_regularizer=l1_l2(ALPHA_REG, ALPHA_REG),)(ret)
    ret = Activation('sigmoid')(ret)
    return multiply([x, ret])


def se_res_block(x, num_filters, strides=(1, 1)):
    """
    A squeeze excitation residual block

    #Arguments
        x: the input layer
        num_filters: number of convolutional filters to use
        strides: the stride to use in the conv layers
    """
    ret = BatchNormalization()(x)
    ret = ReLU()(ret)

    ret = Conv2D(num_filters, kernel_size=(1, 1), kernel_initializer=glorot_normal(SEED), kernel_regularizer=l1_l2(ALPHA_REG, ALPHA_REG), padding='same')(ret)
    ret = BatchNormalization()(ret)
    ret = ReLU()(ret)

    ret = Conv2D(num_filters, kernel_size=(3, 3), kernel_initializer=glorot_normal(SEED), kernel_regularizer=l1_l2(ALPHA_REG, ALPHA_REG), strides=strides, padding='same')(ret)
    ret = BatchNormalization()(ret)
    ret = ReLU()(ret)

    ret = Conv2D(num_filters * BOTTLENECK_RATIO, kernel_initializer=glorot_normal(SEED), kernel_regularizer=l1_l2(ALPHA_REG, ALPHA_REG), kernel_size=(1, 1), padding='same')(ret)

    ret = se_block(ret, num_filters)

    if strides != (1, 1) or x._keras_shape[-1] != num_filters * BOTTLENECK_RATIO:
        ret = add([Conv2D(num_filters * BOTTLENECK_RATIO, kernel_size=(1, 1), kernel_initializer=glorot_normal(SEED), kernel_regularizer=l1_l2(ALPHA_REG, ALPHA_REG), padding='same', strides=strides)(x), ret])
    else:
        ret = add([x, ret])
    return ret


def se_res_model(layers):
    """
    A squeeze-excitation Binary Classification RESnet implementation.

    #Arguments
        layers: a list pertaining to the number of 64, 128, 256, and 512 resnet blocks in the network.

    Returns a squeese excitation network RESnet model defined by the given layer list.
    """
    inp = Input(shape=(IMAGE_DIM, IMAGE_DIM, 1))

    x = Conv2D(64, kernel_size=(3, 3), kernel_initializer=glorot_normal(SEED), kernel_regularizer=l1_l2(ALPHA_REG, ALPHA_REG), padding='same', strides=(2, 2))(inp)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = MaxPooling2D()(x)

    for i in range(layers[0]):
        x = se_res_block(x, 64)

    x = se_res_block(x, 128, strides=(2, 2))
    for i in range(layers[1] - 1):
        x = se_res_block(x, 128)

    x = se_res_block(x, 256, strides=(2, 2))
    for i in range(layers[2] - 1):
        x = se_res_block(x, 256)

    x = se_res_block(x, 512, strides=(2, 2))
    for i in range(layers[3] - 1):
        x = se_res_block(x, 512)

    x = GlobalMaxPooling2D()(x)
    x = Dense(1, kernel_regularizer=l1_l2(ALPHA_REG, ALPHA_REG))(x)
    x = Activation('sigmoid')(x)

    ret = Model(inputs=[inp], outputs=x)
    return ret

def _conv_block(x, num_filters, bottleneck=False):
    ret = BatchNormalization()(x)
    ret = ReLU()(ret)
    if bottleneck:
        ret = Conv2D(num_filters * BOTTLENECK_RATIO, (1, 1), kernel_initializer=glorot_normal(SEED), kernel_regularizer=l1_l2(ALPHA_REG, ALPHA_REG), padding='same')(ret)
        ret = BatchNormalization()(ret)
        ret = ReLU(ret)
    ret = Conv2D(num_filters, (3, 3), kernel_initializer=glorot_normal(SEED), kernel_regularizer=l1_l2(ALPHA_REG, ALPHA_REG), padding='same')(ret)
    return ret

def dense_block(x, num_layers, num_filters, growth_rate, bottleneck=False):
    pass



def train(train_from=None, use_latest=False, recompile=False, new_dir=False):
    """
    Train a model.

    train_from: the filepath to the model to train from, if specified.
    use_latest: whether or not to start training from the last saved model of the same name
    recompile: whether or not to recompile the model before training
    new_dir: whether or not to create a new directory
    """
    global init_epoch

    if train_from:
        curr_model = keras.models.load_model(train_from)
        init_epoch = int(
            re.search('[0-9]+', re.search('weights\.[0-9]+-', train_from).group(0)).group(
                0))
    elif use_latest:
        latest = max(glob.glob(f'Models/{MODEL_NAME}/*.hdf5'), key=os.path.getctime)
        curr_model = keras.models.load_model(latest)
        init_epoch = int(
            re.search('[0-9]+', re.search('weights\.[0-9]+-', latest).group(0)).group(
                0))
    else:
        curr_model = se_res_model([NUM_64_BLOCK, NUM_128_BLOCK, NUM_256_BLOCK, NUM_512_BLOCK])
        curr_model.compile(optimizers.Adam(lr=INIT_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    print(curr_model.summary())

    if new_dir:
        mkdir(f'Models/{MODEL_NAME}/')

    random.seed(SEED)
    np.random.seed(SEED)

    print('Fitting Model')

    if train_from or use_latest:
        history = curr_model.fit_generator(
            aug.flow_from_directory('trainingdata/',
                                    target_size=(IMAGE_DIM, IMAGE_DIM),
                                    color_mode='grayscale',
                                    class_mode='binary',
                                    batch_size=BATCH_SIZE),
            validation_data=aug.flow_from_directory('valdata/',
                                                    target_size=(IMAGE_DIM, IMAGE_DIM),
                                                    color_mode='grayscale',
                                                    class_mode='binary',
                                                    batch_size=BATCH_SIZE),
            validation_steps=NUM_VALIDATION_EXAMPLES / BATCH_SIZE,
            verbose=1,
            epochs=10000,
            callbacks=callbacks,
            steps_per_epoch=NUM_TRAINING_EXAMPLES / (4 * BATCH_SIZE),
            class_weight={v: k for v, k in enumerate(CLASS_WEIGHTS)},
            initial_epoch=init_epoch)
    else:
        history = curr_model.fit_generator(
            aug.flow_from_directory('trainingdata/',
                                    target_size=(IMAGE_DIM, IMAGE_DIM),
                                    color_mode='grayscale',
                                    class_mode='binary',
                                    batch_size=BATCH_SIZE),
            validation_data=aug.flow_from_directory('valdata/',
                                                    target_size=(IMAGE_DIM, IMAGE_DIM),
                                                    color_mode='grayscale',
                                                    class_mode='binary', batch_size=BATCH_SIZE),
            validation_steps=NUM_VALIDATION_EXAMPLES / BATCH_SIZE,
            verbose=1,
            epochs=10000,
            callbacks=callbacks,
            steps_per_epoch=NUM_TRAINING_EXAMPLES / (4 * BATCH_SIZE),
            class_weight={v: k for v, k in enumerate(CLASS_WEIGHTS)})


def build_dataset_directories():
    """Sorts and copies the training and validation data into more ImageGenerator friendly directories"""
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
    """Copy training/validation files from one directory to another in a folder with matching label"""
    working_count = count
    for file in glob.glob(read_directory_path + '*.png'):
        copyfile(file, f'{write_directory_path}/{label}/{working_count}.png')
        working_count += 1
    return working_count


if __name__ == '__main__':
    train(new_dir=True)
