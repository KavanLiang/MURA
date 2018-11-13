import keras
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
import csv
import glob

from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

import tqdm as tqdm

import sys

MODEL_PATH = 'x200-float16-weights-32-128-256-256-512-512-512-512-.21-0.55.hdf5'

IMAGE_DIM = 200

def preprocess_image(filename):
    im = load_img(filename, color_mode='grayscale', target_size=(IMAGE_DIM, IMAGE_DIM), interpolation='lanczos')
    ret = img_to_array(im, dtype='float16')
    return ret

def predict(input_csv_path, output_csv_path):
    model = keras.models.load_model(MODEL_PATH)
    with open(input_csv_path) as input_csv_file:
        with open(output_csv_path, 'w') as output_csv_file:
            input_csv_reader = csv.reader(input_csv_file)
            output_csv_writer = csv.writer(output_csv_file, lineterminator='\n')
            seen = set()
            for line in input_csv_reader:
                current_study_dir = '/'.join(line[0].split('/')[:-1]) + '/'
                if current_study_dir not in seen:
                    seen.add(current_study_dir)
                    y = []
                    for file in glob.glob(current_study_dir + "*.png"):
                        processed_img = preprocess_image(file)
                        y += [model.predict(processed_img.reshape((1,) + processed_img.shape)).item()]
                    output_csv_writer.writerow([current_study_dir, int(round(np.array(y).mean()))])

def test_validation():
    model = keras.models.load_model(MODEL_PATH)
    with open('MURA-v1.1/valid_labeled_studies.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for line in csv_reader:
            label = int(line[1])
            y = []
            for file in glob.glob(line[0] + '*.png'):
                processed_img = preprocess_image(file)
                y += [model.predict(processed_img.reshape((1,) + processed_img.shape)).item()]
            y = int(round(np.array(y).mean()))
            if label == 0:
                fp += (label == y)
                fn += (label == y)
            else:
                tp += (label == y)
                tn += (label == y)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1

def visualize(img_dir):
    model = keras.models.load_model(MODEL_PATH)
    for img_path in glob.glob(img_dir + '*.png'):
        print(f'Working on {img_path}')
        plt.figure()
        f, ax = plt.subplots(1, 2)
        img = load_img(img_path, color_mode='grayscale', target_size=(IMAGE_DIM, IMAGE_DIM), interpolation='lanczos')
        background = img.convert('RGBA')
        img = img_to_array(img) / 255.0
        grads = visualize_saliency(model, -1, filter_indices=None, seed_input=img, backprop_modifier='guided')
        jet_heatmap = np.uint8(cm.jet(grads)[..., :3] * 255)
        ax[0].imshow(overlay(np.array(Image.fromarray(jet_heatmap).convert('RGBA')), np.array(background), alpha=0.5))
        ax[1].imshow(background)
        plt.show()


if __name__ == '__main__':
    visualize('MURA-v1.1/valid/XR_WRIST/patient11267/study1_positive/')