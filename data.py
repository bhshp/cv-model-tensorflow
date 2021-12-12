import pickle
import numpy as np
import cv2

IMAGE_SIZE = 299
DATA_PATH = './data'

def get_data():
    train_images = []
    train_labels = []
    labels = []
    with open(DATA_PATH + '/batches.meta', 'rb') as f:
        labels = pickle.load(f, encoding='bytes')[b'label_names']

    with open(DATA_PATH + '/data_batch_1', 'rb') as f:
        train_batches = pickle.load(f, encoding='bytes')
        print(train_batches.keys())
        print(train_batches[b'data'].shape)
        print(len(train_batches[b'labels']))
        train_images.extend(train_batches[b'data'][:100])
        train_labels.extend(train_batches[b'labels'][:100])
    exit(0)

    def preprocess_images(image):
        return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)

    train_images = [preprocess_images(image) for image in train_images]

    return labels, np.asarray(train_images), np.asarray(train_labels)

get_data()