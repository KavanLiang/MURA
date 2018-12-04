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

MODEL_PATH = 'Models/reg200-float16-weights-c64x3-c128x4-c256x4-c256x4-c512x6-c512x6-d4096x2weights.131-0.52.hdf5'

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
                        y += [model.predict(processed_img.reshape((1,) + processed_img.shape))]
                    output_csv_writer.writerow([current_study_dir, np.array(y).mean(axis=0).argmax()])

def test_validation():
    model = keras.models.load_model(MODEL_PATH)
    with open('MURA-v1.1/valid_labeled_studies.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        agree = 0
        label_p = 0
        predict_p = 0
        label_n = 0
        predict_n = 0
        total = 0
        for line in csv_reader:
            label = int(line[1])
            total += 1
            y = []
            for file in glob.glob(line[0] + '*.png'):
                processed_img = preprocess_image(file)
                y += [model.predict(processed_img.reshape((1,) + processed_img.shape))]
            y = np.array(y).mean(axis=0).argmax()
            if label == 0:
                label_n += 1
            else:
                label_p += 1
            if y == 0:
                predict_n += 1
            else:
                predict_p += 1
            if y == label:
                agree += 1
        p_0 = agree / total
        p_e = ((predict_p / total) * (label_p / total)) + ((predict_n / total) * (label_n / total))
        k = (p_0 - p_e)/(1-p_e)
        return (agree, label_p, predict_p, label_n, predict_n, k)

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
    predict('MURA-v1.1/valid_image_paths.csv', 'out.csv')
    print(test_validation())