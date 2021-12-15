import cv2
import numpy as np
import pickle
import tensorflow as tf
print(tf.__version__)


if tf.__version__.startswith('2.'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
tf.disable_eager_execution()

IMAGE_SIZE = 299
DATA_PATH = './../../data'
INPUT_CKPT_PATH = './model/model.ckpt-1'
OUTPUT_PATH = './result'

def Inception_V3(features, labels, mode):
    input_layer = tf.reshape(features['x'],
                             [-1, IMAGE_SIZE, IMAGE_SIZE, 3],
                             name='data')

    weights = {
        'conv_1': tf.Variable(tf.truncated_normal([3, 3, 3, 32], mean=0, stddev=1e-2)),
        'conv_2': tf.Variable(tf.truncated_normal([3, 3, 32, 32], mean=0, stddev=1e-2)),
        'conv_3': tf.Variable(tf.truncated_normal([3, 3, 32, 64], mean=0, stddev=1e-2)),
        'conv_4': tf.Variable(tf.truncated_normal([1, 1, 64, 80], mean=0, stddev=1e-2)),
        'conv_5': tf.Variable(tf.truncated_normal([3, 3, 80, 192], mean=0, stddev=1e-2)),
        'conv_6_1': tf.Variable(tf.truncated_normal([1, 1, 192, 64], mean=0, stddev=1e-2)),
        'conv_6_2_1': tf.Variable(tf.truncated_normal([1, 1, 192, 48], mean=0, stddev=1e-2)),
        'conv_6_2_2': tf.Variable(tf.truncated_normal([5, 5, 48, 64], mean=0, stddev=1e-2)),
        'conv_6_3_1': tf.Variable(tf.truncated_normal([1, 1, 192, 64], mean=0, stddev=1e-2)),
        'conv_6_3_2': tf.Variable(tf.truncated_normal([3, 3, 64, 96], mean=0, stddev=1e-2)),
        'conv_6_3_3': tf.Variable(tf.truncated_normal([3, 3, 96, 96], mean=0, stddev=1e-2)),
        'conv_6_4_2': tf.Variable(tf.truncated_normal([1, 1, 192, 32], mean=0, stddev=1e-2)),
        'conv_7_1': tf.Variable(tf.truncated_normal([1, 1, 256, 64], mean=0, stddev=1e-2)),
        'conv_7_2_1': tf.Variable(tf.truncated_normal([1, 1, 256, 48], mean=0, stddev=1e-2)),
        'conv_7_2_2': tf.Variable(tf.truncated_normal([5, 5, 48, 64], mean=0, stddev=1e-2)),
        'conv_7_3_1': tf.Variable(tf.truncated_normal([1, 1, 256, 64], mean=0, stddev=1e-2)),
        'conv_7_3_2': tf.Variable(tf.truncated_normal([3, 3, 64, 96], mean=0, stddev=1e-2)),
        'conv_7_3_3': tf.Variable(tf.truncated_normal([3, 3, 96, 96], mean=0, stddev=1e-2)),
        'conv_7_4_2': tf.Variable(tf.truncated_normal([1, 1, 256, 64], mean=0, stddev=1e-2)),
        'conv_8_1': tf.Variable(tf.truncated_normal([1, 1, 288, 64], mean=0, stddev=1e-2)),
        'conv_8_2_1': tf.Variable(tf.truncated_normal([1, 1, 288, 48], mean=0, stddev=1e-2)),
        'conv_8_2_2': tf.Variable(tf.truncated_normal([5, 5, 48, 64], mean=0, stddev=1e-2)),
        'conv_8_3_1': tf.Variable(tf.truncated_normal([1, 1, 288, 64], mean=0, stddev=1e-2)),
        'conv_8_3_2': tf.Variable(tf.truncated_normal([3, 3, 64, 96], mean=0, stddev=1e-2)),
        'conv_8_3_3': tf.Variable(tf.truncated_normal([3, 3, 96, 96], mean=0, stddev=1e-2)),
        'conv_8_4_2': tf.Variable(tf.truncated_normal([1, 1, 288, 64], mean=0, stddev=1e-2)),
        'conv_9_2': tf.Variable(tf.truncated_normal([3, 3, 288, 384], mean=0, stddev=1e-2)),
        'conv_9_3_1': tf.Variable(tf.truncated_normal([1, 1, 288, 64], mean=0, stddev=1e-2)),
        'conv_9_3_2': tf.Variable(tf.truncated_normal([3, 3, 64, 96], mean=0, stddev=1e-2)),
        'conv_9_3_3': tf.Variable(tf.truncated_normal([3, 3, 96, 96], mean=0, stddev=1e-2)),
        'conv_10_1': tf.Variable(tf.truncated_normal([1, 1, 768, 192], mean=0, stddev=1e-2)),
        'conv_10_2_1': tf.Variable(tf.truncated_normal([1, 1, 768, 128], mean=0, stddev=1e-2)),
        'conv_10_2_2': tf.Variable(tf.truncated_normal([1, 7, 128, 128], mean=0, stddev=1e-2)),
        'conv_10_2_3': tf.Variable(tf.truncated_normal([7, 1, 128, 192], mean=0, stddev=1e-2)),
        'conv_10_3_1': tf.Variable(tf.truncated_normal([1, 1, 768, 128], mean=0, stddev=1e-2)),
        'conv_10_3_2': tf.Variable(tf.truncated_normal([7, 1, 128, 128], mean=0, stddev=1e-2)),
        'conv_10_3_3': tf.Variable(tf.truncated_normal([1, 7, 128, 128], mean=0, stddev=1e-2)),
        'conv_10_3_4': tf.Variable(tf.truncated_normal([7, 1, 128, 128], mean=0, stddev=1e-2)),
        'conv_10_3_5': tf.Variable(tf.truncated_normal([1, 7, 128, 192], mean=0, stddev=1e-2)),
        'conv_10_4_2': tf.Variable(tf.truncated_normal([1, 1, 768, 192], mean=0, stddev=1e-2)),
        'conv_11_1': tf.Variable(tf.truncated_normal([1, 1, 768, 192], mean=0, stddev=1e-2)),
        'conv_11_2_1': tf.Variable(tf.truncated_normal([1, 1, 768, 160], mean=0, stddev=1e-2)),
        'conv_11_2_2': tf.Variable(tf.truncated_normal([1, 7, 160, 160], mean=0, stddev=1e-2)),
        'conv_11_2_3': tf.Variable(tf.truncated_normal([7, 1, 160, 192], mean=0, stddev=1e-2)),
        'conv_11_3_1': tf.Variable(tf.truncated_normal([1, 1, 768, 160], mean=0, stddev=1e-2)),
        'conv_11_3_2': tf.Variable(tf.truncated_normal([7, 1, 160, 160], mean=0, stddev=1e-2)),
        'conv_11_3_3': tf.Variable(tf.truncated_normal([1, 7, 160, 160], mean=0, stddev=1e-2)),
        'conv_11_3_4': tf.Variable(tf.truncated_normal([7, 1, 160, 160], mean=0, stddev=1e-2)),
        'conv_11_3_5': tf.Variable(tf.truncated_normal([1, 7, 160, 192], mean=0, stddev=1e-2)),
        'conv_11_4_2': tf.Variable(tf.truncated_normal([1, 1, 768, 192], mean=0, stddev=1e-2)),
        'conv_12_1': tf.Variable(tf.truncated_normal([1, 1, 768, 192], mean=0, stddev=1e-2)),
        'conv_12_2_1': tf.Variable(tf.truncated_normal([1, 1, 768, 160], mean=0, stddev=1e-2)),
        'conv_12_2_2': tf.Variable(tf.truncated_normal([1, 7, 160, 160], mean=0, stddev=1e-2)),
        'conv_12_2_3': tf.Variable(tf.truncated_normal([7, 1, 160, 192], mean=0, stddev=1e-2)),
        'conv_12_3_1': tf.Variable(tf.truncated_normal([1, 1, 768, 160], mean=0, stddev=1e-2)),
        'conv_12_3_2': tf.Variable(tf.truncated_normal([7, 1, 160, 160], mean=0, stddev=1e-2)),
        'conv_12_3_3': tf.Variable(tf.truncated_normal([1, 7, 160, 160], mean=0, stddev=1e-2)),
        'conv_12_3_4': tf.Variable(tf.truncated_normal([7, 1, 160, 160], mean=0, stddev=1e-2)),
        'conv_12_3_5': tf.Variable(tf.truncated_normal([1, 7, 160, 192], mean=0, stddev=1e-2)),
        'conv_12_4_2': tf.Variable(tf.truncated_normal([1, 1, 768, 192], mean=0, stddev=1e-2)),
        'conv_13_1': tf.Variable(tf.truncated_normal([1, 1, 768, 192], mean=0, stddev=1e-2)),
        'conv_13_2_1': tf.Variable(tf.truncated_normal([1, 1, 768, 192], mean=0, stddev=1e-2)),
        'conv_13_2_2': tf.Variable(tf.truncated_normal([1, 7, 192, 192], mean=0, stddev=1e-2)),
        'conv_13_2_3': tf.Variable(tf.truncated_normal([7, 1, 192, 192], mean=0, stddev=1e-2)),
        'conv_13_3_1': tf.Variable(tf.truncated_normal([1, 1, 768, 192], mean=0, stddev=1e-2)),
        'conv_13_3_2': tf.Variable(tf.truncated_normal([7, 1, 192, 192], mean=0, stddev=1e-2)),
        'conv_13_3_3': tf.Variable(tf.truncated_normal([1, 7, 192, 192], mean=0, stddev=1e-2)),
        'conv_13_3_4': tf.Variable(tf.truncated_normal([7, 1, 192, 192], mean=0, stddev=1e-2)),
        'conv_13_3_5': tf.Variable(tf.truncated_normal([1, 7, 192, 192], mean=0, stddev=1e-2)),
        'conv_13_4_2': tf.Variable(tf.truncated_normal([1, 1, 768, 192], mean=0, stddev=1e-2)),
        'conv_14_2_1': tf.Variable(tf.truncated_normal([1, 1, 768, 192], mean=0, stddev=1e-2)),
        'conv_14_2_2': tf.Variable(tf.truncated_normal([3, 3, 192, 320], mean=0, stddev=1e-2)),
        'conv_14_3_1': tf.Variable(tf.truncated_normal([1, 1, 768, 192], mean=0, stddev=1e-2)),
        'conv_14_3_2': tf.Variable(tf.truncated_normal([1, 7, 192, 192], mean=0, stddev=1e-2)),
        'conv_14_3_3': tf.Variable(tf.truncated_normal([7, 1, 192, 192], mean=0, stddev=1e-2)),
        'conv_14_3_4': tf.Variable(tf.truncated_normal([3, 3, 192, 192], mean=0, stddev=1e-2)),
        'conv_15_1': tf.Variable(tf.truncated_normal([1, 1, 1280, 320], mean=0, stddev=1e-2)),
        'conv_15_2_1': tf.Variable(tf.truncated_normal([1, 1, 1280, 384], mean=0, stddev=1e-2)),
        'conv_15_2_2_1': tf.Variable(tf.truncated_normal([1, 3, 384, 384], mean=0, stddev=1e-2)),
        'conv_15_2_2_2': tf.Variable(tf.truncated_normal([3, 1, 384, 384], mean=0, stddev=1e-2)),
        'conv_15_3_1': tf.Variable(tf.truncated_normal([1, 1, 1280, 448], mean=0, stddev=1e-2)),
        'conv_15_3_2': tf.Variable(tf.truncated_normal([3, 3, 448, 384], mean=0, stddev=1e-2)),
        'conv_15_3_3_1': tf.Variable(tf.truncated_normal([1, 3, 384, 384], mean=0, stddev=1e-2)),
        'conv_15_3_3_2': tf.Variable(tf.truncated_normal([3, 1, 384, 384], mean=0, stddev=1e-2)),
        'conv_15_4_2': tf.Variable(tf.truncated_normal([1, 1, 1280, 192], mean=0, stddev=1e-2)),
        'conv_16_1': tf.Variable(tf.truncated_normal([1, 1, 2048, 320], mean=0, stddev=1e-2)),
        'conv_16_2_1': tf.Variable(tf.truncated_normal([1, 1, 2048, 384], mean=0, stddev=1e-2)),
        'conv_16_2_2_1': tf.Variable(tf.truncated_normal([1, 3, 384, 384], mean=0, stddev=1e-2)),
        'conv_16_2_2_2': tf.Variable(tf.truncated_normal([3, 1, 384, 384], mean=0, stddev=1e-2)),
        'conv_16_3_1': tf.Variable(tf.truncated_normal([1, 1, 2048, 448], mean=0, stddev=1e-2)),
        'conv_16_3_2': tf.Variable(tf.truncated_normal([3, 3, 448, 384], mean=0, stddev=1e-2)),
        'conv_16_3_3_1': tf.Variable(tf.truncated_normal([1, 3, 384, 384], mean=0, stddev=1e-2)),
        'conv_16_3_3_2': tf.Variable(tf.truncated_normal([3, 1, 384, 384], mean=0, stddev=1e-2)),
        'conv_16_4_2': tf.Variable(tf.truncated_normal([1, 1, 2048, 192], mean=0, stddev=1e-2)),
        'conv_17': tf.Variable(tf.truncated_normal([1, 1, 2048, 1001], mean=0, stddev=1e-2))
    }

    # 32*3*3*3 valid stride=2
    conv_1 = tf.nn.conv2d(input=input_layer,
                          filters=weights['conv_1'],
                          strides=[1, 2, 2, 1],
                          padding='VALID',
                          name='conv_1')

    relu_1 = tf.nn.relu(features=conv_1,
                        name='relu_1')

    # 32*3*3*32 valid stride=1
    conv_2 = tf.nn.conv2d(input=relu_1,
                          filters=weights['conv_2'],
                          strides=[1, 1, 1, 1],
                          padding='VALID',
                          name='conv_2')

    relu_2 = tf.nn.relu(features=conv_2,
                        name='relu_2')

    # 64*3*3*32 same stride=1
    conv_3 = tf.nn.conv2d(input=relu_2,
                          filters=weights['conv_3'],
                          strides=[1, 1, 1, 1],
                          padding='SAME',
                          name='conv_3')

    relu_3 = tf.nn.relu(features=conv_3,
                        name='relu_3')

    # filter_size=3 valid stride=2
    max_pooling_1 = tf.nn.max_pool2d(input=relu_3,
                                     ksize=3,
                                     strides=2,
                                     padding='VALID',
                                     name='max_pooling_1')

    # 80*1*1*64 valid stride=1
    conv_4 = tf.nn.conv2d(input=max_pooling_1,
                          filters=weights['conv_4'],
                          strides=[1, 1, 1, 1],
                          padding='VALID',
                          name='conv_4')

    relu_4 = tf.nn.relu(features=conv_4,
                        name='relu_4')

    # 192*3*3*80 valid stride=1
    conv_5 = tf.nn.conv2d(input=relu_4,
                          filters=weights['conv_5'],
                          strides=[1, 1, 1, 1],
                          padding='VALID',
                          name='conv_5')

    relu_5 = tf.nn.relu(features=conv_5,
                        name='relu_5')

    # filter_size=3 valid stride=2
    max_pooling_2 = tf.nn.max_pool2d(input=relu_5,
                                     ksize=3,
                                     strides=2,
                                     padding='VALID',
                                     name='max_pooling_2')

    # 64*1*1*192 same stride=1
    conv_6_1 = tf.nn.conv2d(input=max_pooling_2,
                            filters=weights['conv_6_1'],
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                            name='conv_6_1')

    relu_6_1 = tf.nn.relu(features=conv_6_1,
                          name='relu_6_1')

    # 48*1*1*192 same stride=1
    conv_6_2_1 = tf.nn.conv2d(input=max_pooling_2,
                              filters=weights['conv_6_2_1'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_6_2_1')

    relu_6_2_1 = tf.nn.relu(features=conv_6_2_1,
                            name='relu_6_2_1')

    # 64*5*5*48 same stride=1
    conv_6_2_2 = tf.nn.conv2d(input=relu_6_2_1,
                              filters=weights['conv_6_2_2'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_6_2_2')

    relu_6_2_2 = tf.nn.relu(features=conv_6_2_2,
                            name='relu_6_2_2')

    # 64*1*1*192 same stride=1
    conv_6_3_1 = tf.nn.conv2d(input=max_pooling_2,
                              filters=weights['conv_6_3_1'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_6_3_1')

    relu_6_3_1 = tf.nn.relu(features=conv_6_3_1,
                            name='relu_6_3_1')

    # 96*3*3*64 same stride=1
    conv_6_3_2 = tf.nn.conv2d(input=relu_6_3_1,
                              filters=weights['conv_6_3_2'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_6_3_2')

    relu_6_3_2 = tf.nn.relu(features=conv_6_3_2,
                            name='relu_6_3_2')

    # 96*3*3*96 same stride=1
    conv_6_3_3 = tf.nn.conv2d(input=relu_6_3_2,
                              filters=weights['conv_6_3_3'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_6_3_3')

    relu_6_3_3 = tf.nn.relu(features=conv_6_3_3,
                            name='relu_6_3_3')

    # filter_size=3 same stride=1
    average_pool_6_4_1 = tf.nn.avg_pool2d(value=max_pooling_2,
                                          ksize=3,
                                          strides=1,
                                          padding='SAME',
                                          name='average_pool_6_4_1')

    # 32*1*1*192 same stride=1
    conv_6_4_2 = tf.nn.conv2d(input=average_pool_6_4_1,
                              filters=weights['conv_6_4_2'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_6_4_2')

    relu_6_4_2 = tf.nn.relu(features=conv_6_4_2,
                            name='relu_6_4_2')

    # 15
    concat_1 = tf.concat(
        values=[relu_6_1, relu_6_2_2, relu_6_3_3, relu_6_4_2], axis=3, name='concat_1')

    # 64*1*1*256 same stride=1
    conv_7_1 = tf.nn.conv2d(input=concat_1,
                            filters=weights['conv_7_1'],
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                            name='conv_7_1')

    relu_7_1 = tf.nn.relu(features=conv_7_1,
                          name='relu_7_1')

    # 48*1*1*256 same stride=1
    conv_7_2_1 = tf.nn.conv2d(input=concat_1,
                              filters=weights['conv_7_2_1'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_7_2_1')

    relu_7_2_1 = tf.nn.relu(features=conv_7_2_1,
                            name='relu_7_2_1')

    # 64*5*5*48 same stride=1
    conv_7_2_2 = tf.nn.conv2d(input=relu_7_2_1,
                              filters=weights['conv_7_2_2'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_7_2_2')

    relu_7_2_2 = tf.nn.relu(features=conv_7_2_2,
                            name='relu_7_2_2')

    # 64*1*1*256 same stride=1
    conv_7_3_1 = tf.nn.conv2d(input=concat_1,
                              filters=weights['conv_7_3_1'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_7_3_1')

    relu_7_3_1 = tf.nn.relu(features=conv_7_3_1,
                            name='relu_7_3_1')

    # 96*3*3*64 same stride=1
    conv_7_3_2 = tf.nn.conv2d(input=relu_7_3_1,
                              filters=weights['conv_7_3_2'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_7_3_2')

    relu_7_3_2 = tf.nn.relu(features=conv_7_3_2,
                            name='relu_7_3_2')

    # 96*3*3*96 same stride=1
    conv_7_3_3 = tf.nn.conv2d(input=relu_7_3_2,
                              filters=weights['conv_7_3_3'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_7_3_3')

    relu_7_3_3 = tf.nn.relu(features=conv_7_3_3,
                            name='relu_7_3_3')

    # filter_size=3 same stride=1
    average_pool_7_4_1 = tf.nn.avg_pool2d(value=concat_1,
                                          ksize=3,
                                          strides=1,
                                          padding='SAME',
                                          name='average_pool_7_4_1')

    # 64*1*1*256 same stride=1
    conv_7_4_2 = tf.nn.conv2d(input=average_pool_7_4_1,
                              filters=weights['conv_7_4_2'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_7_4_2')

    relu_7_4_2 = tf.nn.relu(features=conv_7_4_2,
                            name='relu_7_4_2')

    # 24
    concat_2 = tf.concat(
        values=[relu_7_1, relu_7_2_2, relu_7_3_3, relu_7_4_2], axis=3, name='concat_2')

    # 64*1*1*288 same stride=1
    conv_8_1 = tf.nn.conv2d(input=concat_2,
                            filters=weights['conv_8_1'],
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                            name='conv_8_1')

    relu_8_1 = tf.nn.relu(features=conv_8_1,
                          name='relu_8_1')

    # 48*1*1*288 same stride=1
    conv_8_2_1 = tf.nn.conv2d(input=concat_2,
                              filters=weights['conv_8_2_1'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_8_2_1')

    relu_8_2_1 = tf.nn.relu(features=conv_8_2_1,
                            name='relu_8_2_1')

    # 64*5*5*48 same stride=1
    conv_8_2_2 = tf.nn.conv2d(input=relu_8_2_1,
                              filters=weights['conv_8_2_2'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_8_2_2')

    relu_8_2_2 = tf.nn.relu(features=conv_8_2_2,
                            name='relu_8_2_2')

    # 64*1*1*288 same stride=1
    conv_8_3_1 = tf.nn.conv2d(input=concat_2,
                              filters=weights['conv_8_3_1'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_8_3_1')

    relu_8_3_1 = tf.nn.relu(features=conv_8_3_1,
                            name='relu_8_3_1')

    # 96*3*3*64 same stride=1
    conv_8_3_2 = tf.nn.conv2d(input=relu_8_3_1,
                              filters=weights['conv_8_3_2'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_8_3_2')

    relu_8_3_2 = tf.nn.relu(features=conv_8_3_2,
                            name='relu_8_3_2')

    # 96*3*3*96 same stride=1
    conv_8_3_3 = tf.nn.conv2d(input=relu_8_3_2,
                              filters=weights['conv_8_3_3'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_8_3_3')

    relu_8_3_3 = tf.nn.relu(features=conv_8_3_3,
                            name='relu_8_3_3')

    # filter_size=3 same stride=1
    average_pool_8_4_1 = tf.nn.avg_pool2d(value=concat_2,
                                          ksize=3,
                                          strides=1,
                                          padding='SAME',
                                          name='average_pool_8_4_1')

    # 64*1*1*288 same stride=1
    conv_8_4_2 = tf.nn.conv2d(input=average_pool_8_4_1,
                              filters=weights['conv_8_4_2'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_8_4_2')

    relu_8_4_2 = tf.nn.relu(features=conv_8_4_2,
                            name='relu_8_4_2')

    # 33
    concat_3 = tf.concat(
        values=[relu_8_1, relu_8_2_2, relu_8_3_3, relu_8_4_2], axis=3, name='concat_2')

    # filter_size=3 valid stride=2
    max_pooling_9_1 = tf.nn.max_pool2d(input=concat_3,
                                       ksize=[1, 3, 3, 1],
                                       strides=[1, 2, 2, 1],
                                       padding='VALID',
                                       name='max_pooling_9_1')

    # 384*3*3*288 valid stride=2
    conv_9_2 = tf.nn.conv2d(input=concat_3,
                            filters=weights['conv_9_2'],
                            strides=[1, 2, 2, 1],
                            padding='VALID',
                            name='conv_9_2')

    relu_9_2 = tf.nn.relu(features=conv_9_2,
                          name='relu_9_2')

    # 64*1*1*288 same stride=1
    conv_9_3_1 = tf.nn.conv2d(input=concat_3,
                              filters=weights['conv_9_3_1'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_9_3_1')

    relu_9_3_1 = tf.nn.relu(features=conv_9_3_1,
                            name='relu_9_3_1')

    # 96*3*3*64 same stride=1
    conv_9_3_2 = tf.nn.conv2d(input=relu_9_3_1,
                              filters=weights['conv_9_3_2'],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              name='conv_9_3_2')

    relu_9_3_2 = tf.nn.relu(features=conv_9_3_2,
                            name='relu_9_3_2')

    # 96*3*3*96 valid stride=2
    conv_9_3_3 = tf.nn.conv2d(input=relu_9_3_2,
                              filters=weights['conv_9_3_3'],
                              strides=[1, 2, 2, 1],
                              padding='VALID',
                              name='conv_9_3_3')

    relu_9_3_3 = tf.nn.relu(features=conv_9_3_3,
                            name='relu_9_3_3')

    # 39
    concat_4 = tf.concat(
        values=[max_pooling_9_1, relu_9_2, relu_9_3_3], axis=3, name='concat_4')

    # 192*1*1*768 same stride=1
    conv_10_1 = tf.nn.conv2d(input=concat_4,
                             filters=weights['conv_10_1'],
                             strides=[1, 1, 1, 1],
                             padding='SAME',
                             name='conv_10_1')

    relu_10_1 = tf.nn.relu(features=conv_10_1,
                           name='relu_10_1')

    # 128*1*1*768 same stride=1
    conv_10_2_1 = tf.nn.conv2d(input=concat_4,
                               filters=weights['conv_10_2_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_10_2_1')

    relu_10_2_1 = tf.nn.relu(features=conv_10_2_1,
                             name='relu_10_2_1')

    # 128*1*7*128 same stride=1
    conv_10_2_2 = tf.nn.conv2d(input=relu_10_2_1,
                               filters=weights['conv_10_2_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_10_2_2')

    relu_10_2_2 = tf.nn.relu(features=conv_10_2_2,
                             name='relu_10_2_2')

    # 192*7*1*128 same stride=1
    conv_10_2_3 = tf.nn.conv2d(input=relu_10_2_2,
                               filters=weights['conv_10_2_3'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_10_2_3')

    relu_10_2_3 = tf.nn.relu(features=conv_10_2_3,
                             name='relu_10_2_3')

    # 128*1*1*768 same stride=1
    conv_10_3_1 = tf.nn.conv2d(input=concat_4,
                               filters=weights['conv_10_3_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_10_3_1')

    relu_10_3_1 = tf.nn.relu(features=conv_10_3_1,
                             name='relu_10_3_1')

    # 128*7*1*128 same stride=1
    conv_10_3_2 = tf.nn.conv2d(input=relu_10_3_1,
                               filters=weights['conv_10_3_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_10_3_2')

    relu_10_3_2 = tf.nn.relu(features=conv_10_3_2,
                             name='relu_10_3_2')

    # 128*1*7*128 same stride=1
    conv_10_3_3 = tf.nn.conv2d(input=relu_10_3_2,
                               filters=weights['conv_10_3_3'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_10_3_3')

    relu_10_3_3 = tf.nn.relu(features=conv_10_3_3,
                             name='relu_10_3_3')

    # 128*7*1*128 same stride=1
    conv_10_3_4 = tf.nn.conv2d(input=relu_10_3_3,
                               filters=weights['conv_10_3_4'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_10_3_4')

    relu_10_3_4 = tf.nn.relu(features=conv_10_3_4,
                             name='relu_10_3_4')

    # 192*1*7*128 same stride=1
    conv_10_3_5 = tf.nn.conv2d(input=relu_10_3_4,
                               filters=weights['conv_10_3_5'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_10_3_5')

    relu_10_3_5 = tf.nn.relu(features=conv_10_3_5,
                             name='relu_10_3_5')

    # filter_size=3 same strides=1
    average_pool_10_4_1 = tf.nn.avg_pool(value=concat_4,
                                         ksize=[1, 3, 3, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME',
                                         name='average_pool_10_4_1')

    # 192*1*1*768 same stride=1
    conv_10_4_2 = tf.nn.conv2d(input=average_pool_10_4_1,
                               filters=weights['conv_10_4_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_10_4_2')

    relu_10_4_2 = tf.nn.relu(features=conv_10_4_2,
                             name='relu_10_4_2')

    # 51
    concat_5 = tf.concat(
        values=[relu_10_1, relu_10_2_3, relu_10_3_5, relu_10_4_2], axis=3, name='concat_5')

    # 192*1*1*768 same stride=1
    conv_11_1 = tf.nn.conv2d(input=concat_5,
                             filters=weights['conv_11_1'],
                             strides=[1, 1, 1, 1],
                             padding='SAME',
                             name='conv_11_1')

    relu_11_1 = tf.nn.relu(features=conv_11_1,
                           name='relu_11_1')

    # 160*1*1*768 same stride=1
    conv_11_2_1 = tf.nn.conv2d(input=concat_5,
                               filters=weights['conv_11_2_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_11_2_1')

    relu_11_2_1 = tf.nn.relu(features=conv_11_2_1,
                             name='relu_11_2_1')

    # 160*1*7*160 same stride=1
    conv_11_2_2 = tf.nn.conv2d(input=relu_11_2_1,
                               filters=weights['conv_11_2_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_11_2_2')

    relu_11_2_2 = tf.nn.relu(features=conv_11_2_2,
                             name='relu_11_2_2')

    # 192*7*1*160 same stride=1
    conv_11_2_3 = tf.nn.conv2d(input=relu_11_2_2,
                               filters=weights['conv_11_2_3'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_11_2_3')

    relu_11_2_3 = tf.nn.relu(features=conv_11_2_3,
                             name='relu_11_2_3')

    # 160*1*1*768 same stride=1
    conv_11_3_1 = tf.nn.conv2d(input=concat_5,
                               filters=weights['conv_11_3_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_11_3_1')

    relu_11_3_1 = tf.nn.relu(features=conv_11_3_1,
                             name='relu_11_3_1')

    # 160*7*1*160 same stride=1
    conv_11_3_2 = tf.nn.conv2d(input=relu_11_3_1,
                               filters=weights['conv_11_3_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_11_3_2')

    relu_11_3_2 = tf.nn.relu(features=conv_11_3_2,
                             name='relu_11_3_2')

    # 160*1*7*160 same stride=1
    conv_11_3_3 = tf.nn.conv2d(input=relu_11_3_2,
                               filters=weights['conv_11_3_3'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_11_3_3')

    relu_11_3_3 = tf.nn.relu(features=conv_11_3_3,
                             name='relu_11_3_3')

    # 160*7*1*160 same stride=1
    conv_11_3_4 = tf.nn.conv2d(input=relu_11_3_3,
                               filters=weights['conv_11_3_4'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_11_3_4')

    relu_11_3_4 = tf.nn.relu(features=conv_11_3_4,
                             name='relu_11_3_4')

    # 192*1*7*160 same stride=1
    conv_11_3_5 = tf.nn.conv2d(input=relu_11_3_4,
                               filters=weights['conv_11_3_5'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_11_3_5')

    relu_11_3_5 = tf.nn.relu(features=conv_11_3_5,
                             name='relu_11_3_5')

    # filter_size=3 same strides=1
    average_pool_11_4_1 = tf.nn.avg_pool(value=concat_5,
                                         ksize=[1, 3, 3, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME',
                                         name='average_pool_11_4_1')

    # 192*1*1*768 same stride=1
    conv_11_4_2 = tf.nn.conv2d(input=average_pool_11_4_1,
                               filters=weights['conv_11_4_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_11_4_2')

    relu_11_4_2 = tf.nn.relu(features=conv_11_4_2,
                             name='relu_11_4_2')

    # 63
    concat_6 = tf.concat(
        values=[relu_11_1, relu_11_2_3, relu_11_3_5, relu_11_4_2], axis=3, name='concat_6')

    # 192*1*1*768 same stride=1
    conv_12_1 = tf.nn.conv2d(input=concat_6,
                             filters=weights['conv_12_1'],
                             strides=[1, 1, 1, 1],
                             padding='SAME',
                             name='conv_12_1')

    relu_12_1 = tf.nn.relu(features=conv_12_1,
                           name='relu_12_1')

    # 160*1*1*768 same stride=1
    conv_12_2_1 = tf.nn.conv2d(input=concat_6,
                               filters=weights['conv_12_2_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_12_2_1')

    relu_12_2_1 = tf.nn.relu(features=conv_12_2_1,
                             name='relu_12_2_1')

    # 160*1*7*160 same stride=1
    conv_12_2_2 = tf.nn.conv2d(input=relu_12_2_1,
                               filters=weights['conv_12_2_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_12_2_2')

    relu_12_2_2 = tf.nn.relu(features=conv_12_2_2,
                             name='relu_12_2_2')

    # 192*7*1*160 same stride=1
    conv_12_2_3 = tf.nn.conv2d(input=relu_12_2_2,
                               filters=weights['conv_12_2_3'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_12_2_3')

    relu_12_2_3 = tf.nn.relu(features=conv_12_2_3,
                             name='relu_12_2_3')

    # 160*1*1*768 same stride=1
    conv_12_3_1 = tf.nn.conv2d(input=concat_6,
                               filters=weights['conv_12_3_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_12_3_1')

    relu_12_3_1 = tf.nn.relu(features=conv_12_3_1,
                             name='relu_12_3_1')

    # 160*7*1*160 same stride=1
    conv_12_3_2 = tf.nn.conv2d(input=relu_12_3_1,
                               filters=weights['conv_12_3_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_12_3_2')

    relu_12_3_2 = tf.nn.relu(features=conv_12_3_2,
                             name='relu_12_3_2')

    # 160*1*7*160 same stride=1
    conv_12_3_3 = tf.nn.conv2d(input=relu_12_3_2,
                               filters=weights['conv_12_3_3'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_12_3_3')

    relu_12_3_3 = tf.nn.relu(features=conv_12_3_3,
                             name='relu_12_3_3')

    # 160*7*1*160 same stride=1
    conv_12_3_4 = tf.nn.conv2d(input=relu_12_3_3,
                               filters=weights['conv_12_3_4'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_12_3_4')

    relu_12_3_4 = tf.nn.relu(features=conv_12_3_4,
                             name='relu_12_3_4')

    # 192*1*7*160 same stride=1
    conv_12_3_5 = tf.nn.conv2d(input=relu_12_3_4,
                               filters=weights['conv_12_3_5'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_12_3_5')

    relu_12_3_5 = tf.nn.relu(features=conv_12_3_5,
                             name='relu_12_3_5')

    # filter_size=3 same strides=1
    average_pool_12_4_1 = tf.nn.avg_pool(value=concat_6,
                                         ksize=[1, 3, 3, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME',
                                         name='average_pool_12_4_1')

    # 192*1*1*768 same stride=1
    conv_12_4_2 = tf.nn.conv2d(input=average_pool_12_4_1,
                               filters=weights['conv_12_4_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_12_4_2')

    relu_12_4_2 = tf.nn.relu(features=conv_12_4_2,
                             name='relu_12_4_2')

    # 75
    concat_7 = tf.concat(
        values=[relu_12_1, relu_12_2_3, relu_12_3_5, relu_12_4_2], axis=3, name='concat_6')

    # 192*1*1*768 same stride=1
    conv_13_1 = tf.nn.conv2d(input=concat_7,
                             filters=weights['conv_13_1'],
                             strides=[1, 1, 1, 1],
                             padding='SAME',
                             name='conv_13_1')

    relu_13_1 = tf.nn.relu(features=conv_13_1,
                           name='relu_13_1')

    # 192*1*1*768 same stride=1
    conv_13_2_1 = tf.nn.conv2d(input=concat_7,
                               filters=weights['conv_13_2_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_13_2_1')

    relu_13_2_1 = tf.nn.relu(features=conv_13_2_1,
                             name='relu_13_2_1')

    # 192*1*7*192 same stride=1
    conv_13_2_2 = tf.nn.conv2d(input=relu_13_2_1,
                               filters=weights['conv_13_2_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_13_2_2')

    relu_13_2_2 = tf.nn.relu(features=conv_13_2_2,
                             name='relu_13_2_2')

    # 192*7*1*192 same stride=1
    conv_13_2_3 = tf.nn.conv2d(input=relu_13_2_2,
                               filters=weights['conv_13_2_3'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_13_2_3')

    relu_13_2_3 = tf.nn.relu(features=conv_13_2_3,
                             name='relu_13_2_3')

    # 192*1*1*768 same stride=1
    conv_13_3_1 = tf.nn.conv2d(input=concat_7,
                               filters=weights['conv_13_3_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_13_3_1')

    relu_13_3_1 = tf.nn.relu(features=conv_13_3_1,
                             name='relu_13_3_1')

    # 192*7*1*192 same stride=1
    conv_13_3_2 = tf.nn.conv2d(input=relu_13_3_1,
                               filters=weights['conv_13_3_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_13_3_2')

    relu_13_3_2 = tf.nn.relu(features=conv_13_3_2,
                             name='relu_13_3_2')

    # 192*1*7*192 same stride=1
    conv_13_3_3 = tf.nn.conv2d(input=relu_13_3_2,
                               filters=weights['conv_13_3_3'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_13_3_3')

    relu_13_3_3 = tf.nn.relu(features=conv_13_3_3,
                             name='relu_13_3_3')

    # 192*7*1*192 same stride=1
    conv_13_3_4 = tf.nn.conv2d(input=relu_13_3_3,
                               filters=weights['conv_13_3_4'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_13_3_4')

    relu_13_3_4 = tf.nn.relu(features=conv_13_3_4,
                             name='relu_13_3_4')

    # 192*1*7*192 same stride=1
    conv_13_3_5 = tf.nn.conv2d(input=relu_13_3_4,
                               filters=weights['conv_13_3_5'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_13_3_5')

    relu_13_3_5 = tf.nn.relu(features=conv_13_3_5,
                             name='relu_13_3_5')

    # filter_size=3 same strides=1
    average_pool_13_4_1 = tf.nn.avg_pool(value=concat_7,
                                         ksize=[1, 3, 3, 1],
                                         strides=[1, 1, 1, 1],
                                         padding='SAME',
                                         name='average_pool_13_4_1')

    # 192*1*1*768 same stride=1
    conv_13_4_2 = tf.nn.conv2d(input=average_pool_13_4_1,
                               filters=weights['conv_13_4_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_13_4_2')

    relu_13_4_2 = tf.nn.relu(features=conv_13_4_2,
                             name='relu_13_4_2')

    # 87
    concat_8 = tf.concat(
        values=[relu_13_1, relu_13_2_3, relu_13_3_5, relu_13_4_2], axis=3, name='concat_7')

    # filter_size=3 valid stride=2
    max_pooling_14_1 = tf.nn.max_pool2d(input=concat_8,
                                        ksize=[1, 3, 3, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='VALID',
                                        name='max_pooling_14_1')

    # 192*1*1*768 same stride=1
    conv_14_2_1 = tf.nn.conv2d(input=concat_8,
                               filters=weights['conv_14_2_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_14_2_1')

    relu_14_2_1 = tf.nn.relu(features=conv_14_2_1,
                             name='relu_14_2_1')

    # 320*3*3*192 valid stride=1
    conv_14_2_2 = tf.nn.conv2d(input=relu_14_2_1,
                               filters=weights['conv_14_2_2'],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='conv_14_2_2')

    relu_14_2_2 = tf.nn.relu(features=conv_14_2_2,
                           name='relu_14_2_2')

    # 192*1*1*768 same stride=1
    conv_14_3_1 = tf.nn.conv2d(input=concat_8,
                               filters=weights['conv_14_3_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_14_3_1')

    relu_14_3_1 = tf.nn.relu(features=conv_14_3_1,
                             name='relu_14_3_1')

    # 192*1*7*192 same stride=1
    conv_14_3_2 = tf.nn.conv2d(input=relu_14_3_1,
                               filters=weights['conv_14_3_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_14_3_2')

    relu_14_3_2 = tf.nn.relu(features=conv_14_3_2,
                             name='relu_14_3_2')

    # 192*7*1*192 same stride=1
    conv_14_3_3 = tf.nn.conv2d(input=relu_14_3_2,
                               filters=weights['conv_14_3_3'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_14_3_3')

    relu_14_3_3 = tf.nn.relu(features=conv_14_3_3,
                             name='relu_14_3_3')

    # 192*3*3*192 valid stride=2
    conv_14_3_4 = tf.nn.conv2d(input=relu_14_3_3,
                               filters=weights['conv_14_3_4'],
                               strides=[1, 2, 2, 1],
                               padding='VALID',
                               name='conv_14_3_4')

    relu_14_3_4 = tf.nn.relu(features=conv_14_3_4,
                             name='relu_14_3_4')

    # 95
    concat_9 = tf.concat(
        values=[max_pooling_14_1, relu_14_2, relu_14_3_4], axis=3, name='concat_8')

    # 320*1*1*1280 same stride=1
    conv_15_1 = tf.nn.conv2d(input=concat_9,
                             filters=weights['conv_15_1'],
                             strides=[1, 1, 1, 1],
                             padding='SAME',
                             name='conv_15_1')

    relu_15_1 = tf.nn.relu(features=conv_15_1,
                           name='relu_15_1')

    # 384*1*1*1280 same stride=1
    conv_15_2_1 = tf.nn.conv2d(input=concat_9,
                               filters=weights['conv_15_2_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_15_2_1')

    relu_15_2_1 = tf.nn.relu(features=conv_15_2_1,
                             name='relu_15_2_1')

    # 384*1*3*384 same stride=1
    conv_15_2_2_1 = tf.nn.conv2d(input=relu_15_2_1,
                                 filters=weights['conv_15_2_2_1'],
                                 strides=[1, 1, 1, 1],
                                 padding='SAME',
                                 name='conv_15_2_2_1')

    relu_15_2_2_1 = tf.nn.relu(features=conv_15_2_2_1,
                               name='relu_15_2_2_1')
    # 384*3*1*384 same stride=1
    conv_15_2_2_2 = tf.nn.conv2d(input=relu_15_2_1,
                                 filters=weights['conv_15_2_2_2'],
                                 strides=[1, 1, 1, 1],
                                 padding='SAME',
                                 name='conv_15_2_2_2')

    relu_15_2_2_2 = tf.nn.relu(features=conv_15_2_2_2,
                               name='relu_15_2_2_2')

    concat_15_2_3 = tf.concat(
        values=[relu_15_2_2_1, relu_15_2_2_2], axis=3, name='concat_15_2_3')

    # 448*1*1*1280 same stride=1
    conv_15_3_1 = tf.nn.conv2d(input=concat_9,
                               filters=weights['conv_15_3_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_15_3_1')

    relu_15_3_1 = tf.nn.relu(features=conv_15_3_1,
                             name='relu_15_3_1')

    # 384*3*3*448 same stride=1
    conv_15_3_2 = tf.nn.conv2d(input=relu_15_3_1,
                               filters=weights['conv_15_3_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_15_3_2')

    relu_15_3_2 = tf.nn.relu(features=conv_15_3_2,
                             name='relu_15_3_2')

    # 384*1*3*384 same stride=1
    conv_15_3_3_1 = tf.nn.conv2d(input=relu_15_3_2,
                                 filters=weights['conv_15_3_3_1'],
                                 strides=[1, 1, 1, 1],
                                 padding='SAME',
                                 name='conv_15_3_3_1')

    relu_15_3_3_1 = tf.nn.relu(features=conv_15_3_3_1,
                               name='relu_15_3_3_1')

    # 384*3*1*384 same stride=1
    conv_15_3_3_2 = tf.nn.conv2d(input=relu_15_3_2,
                                 filters=weights['conv_15_3_3_2'],
                                 strides=[1, 1, 1, 1],
                                 padding='SAME',
                                 name='conv_15_3_3_2')

    relu_15_3_3_2 = tf.nn.relu(features=conv_15_3_3_2,
                               name='relu_15_3_3_2')

    concat_15_3_4 = tf.concat(
        values=[relu_15_3_3_1, relu_15_3_3_2], axis=3, name='concat_15_3_4')

    # filter_size=3 same stride=1
    average_pool_15_4_1 = tf.nn.avg_pool2d(value=concat_9,
                                           ksize=3,
                                           strides=[1, 1, 1, 1],
                                           padding='SAME',
                                           name='average_pool_15_4_1')

    # 192*1*1*1280 same stride=1
    conv_15_4_2 = tf.nn.conv2d(input=average_pool_15_4_1,
                               filters=weights['conv_15_4_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_15_4_2')

    relu_15_4_2 = tf.nn.relu(features=conv_15_4_2,
                             name='relu_15_4_2')

    # 108
    concat_10 = tf.concat(
        values=[relu_15_1, concat_15_2_3, concat_15_3_4, relu_15_4_2], axis=3, name='concat_10')

    # 320*1*1*2048 same stride=1
    conv_16_1 = tf.nn.conv2d(input=concat_10,
                             filters=weights['conv_16_1'],
                             strides=[1, 1, 1, 1],
                             padding='SAME',
                             name='conv_16_1')

    relu_16_1 = tf.nn.relu(features=conv_16_1,
                           name='relu_16_1')

    # 384*1*1*2048 same stride=1
    conv_16_2_1 = tf.nn.conv2d(input=concat_10,
                               filters=weights['conv_16_2_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_16_2_1')

    relu_16_2_1 = tf.nn.relu(features=conv_16_2_1,
                             name='relu_16_2_1')

    # 384*1*3*384 same stride=1
    conv_16_2_2_1 = tf.nn.conv2d(input=relu_16_2_1,
                                 filters=weights['conv_16_2_2_1'],
                                 strides=[1, 1, 1, 1],
                                 padding='SAME',
                                 name='conv_16_2_2_1')

    relu_16_2_2_1 = tf.nn.relu(features=conv_16_2_2_1,
                               name='relu_16_2_2_1')
    # 384*3*1*384 same stride=1
    conv_16_2_2_2 = tf.nn.conv2d(input=relu_16_2_1,
                                 filters=weights['conv_16_2_2_2'],
                                 strides=[1, 1, 1, 1],
                                 padding='SAME',
                                 name='conv_16_2_2_2')

    relu_16_2_2_2 = tf.nn.relu(features=conv_16_2_2_2,
                               name='relu_16_2_2_2')

    concat_16_2_3 = tf.concat(
        values=[relu_16_2_2_1, relu_16_2_2_2], axis=3, name='concat_16_2_3')

    # 448*1*1*2048 same stride=1
    conv_16_3_1 = tf.nn.conv2d(input=concat_10,
                               filters=weights['conv_16_3_1'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_16_3_1')

    relu_16_3_1 = tf.nn.relu(features=conv_16_3_1,
                             name='relu_16_3_1')

    # 384*3*3*448 same stride=1
    conv_16_3_2 = tf.nn.conv2d(input=relu_16_3_1,
                               filters=weights['conv_16_3_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_16_3_2')

    relu_16_3_2 = tf.nn.relu(features=conv_16_3_2,
                             name='relu_16_3_2')

    # 384*1*3*384 same stride=1
    conv_16_3_3_1 = tf.nn.conv2d(input=relu_16_3_2,
                                 filters=weights['conv_16_3_3_1'],
                                 strides=[1, 1, 1, 1],
                                 padding='SAME',
                                 name='conv_16_3_3_1')

    relu_16_3_3_1 = tf.nn.relu(features=conv_16_3_3_1,
                               name='relu_16_3_3_1')

    # 384*3*1*384 same stride=1
    conv_16_3_3_2 = tf.nn.conv2d(input=relu_16_3_2,
                                 filters=weights['conv_16_3_3_2'],
                                 strides=[1, 1, 1, 1],
                                 padding='SAME',
                                 name='conv_16_3_3_2')

    relu_16_3_3_2 = tf.nn.relu(features=conv_16_3_3_2,
                               name='relu_16_3_3_2')

    concat_16_3_4 = tf.concat(
        values=[relu_16_3_3_1, relu_16_3_3_2], axis=3, name='concat_16_3_4')

    # filter_size=3 same stride=1
    average_pool_16_4_1 = tf.nn.avg_pool2d(value=concat_10,
                                           ksize=3,
                                           strides=[1, 1, 1, 1],
                                           padding='SAME',
                                           name='average_pool_16_4_1')

    # 192*1*1*2048 same stride=1
    conv_16_4_2 = tf.nn.conv2d(input=average_pool_16_4_1,
                               filters=weights['conv_16_4_2'],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               name='conv_16_4_2')

    relu_16_4_2 = tf.nn.relu(features=conv_16_4_2,
                             name='relu_16_4_2')

    # 121
    concat_11 = tf.concat(
        values=[relu_16_1, concat_16_2_3, concat_16_3_4, relu_16_4_2], axis=3, name='concat_11')

    # filter_size=8 valid stride=2
    average_pool_17 = tf.nn.avg_pool2d(value=concat_11,
                                       ksize=8,
                                       strides=[1, 2, 2, 1],
                                       padding='VALID',
                                       name='average_pool_17')

    # 1001*1*1*2048
    conv_17 = tf.nn.conv2d(input=average_pool_17,
                           filters=weights['conv_17'],
                           strides=[1, 1, 1, 1],
                           padding='SAME',
                           name='conv_17')

    reshape = tf.reshape(conv_17,
                         [-1, 1001],
                         name='reshape')

    logits = reshape

    predictions = {
        'classes': tf.argmax(input=logits, axis=1)
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes'])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def get_data():
    train_images = []
    train_labels = []
    labels = []
    with open(DATA_PATH + '/batches.meta', 'rb') as f:
        labels = pickle.load(f, encoding='bytes')[b'label_names']

    with open(DATA_PATH + '/data_batch_1', 'rb') as f:
        train_batches = pickle.load(f, encoding='bytes')
        train_images.extend(train_batches[b'data'][:20])
        train_labels.extend(train_batches[b'labels'][:20])

    def preprocess_images(image):
        image.resize(32, 32, 3)
        return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0

    train_images = [preprocess_images(image) for image in train_images]

    return labels, np.asarray(train_images), np.asarray(train_labels)


labels, train_images, train_labels = get_data()

eval_images, eval_labels = train_images[:5], train_labels[:5]
train_images, train_labels = train_images[5:], train_labels[5:]

classifier = tf.estimator.Estimator(
    model_fn=Inception_V3, model_dir='./model'
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': train_images},
    y=train_labels,
    batch_size=1,
    num_epochs=1,
    shuffle=False
)
with tf.Graph().as_default():
    classifier.train(input_fn=train_input_fn, steps=1)

print('train done', flush=True)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': eval_images},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results = classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)

saver = tf.train.import_meta_graph(
    INPUT_CKPT_PATH + '.meta', clear_devices=True)
graph = tf.get_default_graph()
input_graph_def = graph.as_graph_def()

with tf.Session() as sess:
    saver.restore(sess, INPUT_CKPT_PATH)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=input_graph_def,
        output_node_names=['reshape']
    )
    import os
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    with tf.gfile.GFile(OUTPUT_PATH + '/frozen.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
