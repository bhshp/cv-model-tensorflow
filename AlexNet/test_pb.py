import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.python.platform import gfile


if tf.__version__.startswith('2.'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
tf.disable_eager_execution()

DATA_PATH = './data/cifar-10-batches-py'
IMAGE_SIZE = 115
OUTPUT_PATH = './result'


def get_data():
    labels = []
    with open(DATA_PATH + '/batches.meta', 'rb') as f:
        labels = pickle.load(f, encoding='bytes')[b'label_names']

    def preprocess_images(image):
        image.resize((32, 32, 3))
        # .reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
        return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)

    with open(DATA_PATH + '/test_batch', 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
        test_images, test_labels = test_data[b'data'][:500], test_data[b'labels'][:500]

    test_images = [preprocess_images(image) for image in test_images]

    return labels, np.asarray(test_images), test_labels


labels, test_images, test_labels = get_data()

sess = tf.Session()
with gfile.GFile(OUTPUT_PATH + '/frozen.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

sess.run(tf.global_variables_initializer())

print([op.name for op in sess.graph.get_operations()])

data = sess.graph.get_tensor_by_name('data:0')

op = sess.graph.get_tensor_by_name('softmax:0')

res = np.argmax(sess.run(op, feed_dict={data: test_images}), axis=1)

cnt = 0
for i in range(len(test_labels)):
    cnt += 1 if res[i] == test_labels[i] else 0
print('{} / {} = {}%'.format(cnt, len(test_labels), 100 * cnt / len(test_labels)))
