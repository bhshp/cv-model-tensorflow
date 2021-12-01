# %tensorflow_version 1.x
import cv2
import numpy as np
import pickle
import tensorflow as tf

print(tf.__version__)

if tf.__version__.startswith('2.'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.INFO)
tf.disable_eager_execution()

IMAGE_SIZE = 115
# DATA_PATH = '/content/drive/MyDrive/cifar-10-batches-py'
DATA_PATH = './data/cifar-10-batches-py'
INPUT_CKPT_PATH = './one_branch_model/model.ckpt-1'
OUTPUT_PATH = './one_branch_result'
# default padding valid


def AlexNet(features, labels, mode):
    weights = {
        # y
        'wc1': tf.Variable(tf.truncated_normal([7, 7, 3, 96],     stddev=0.01), name='wc1'),
        'wc2': tf.Variable(tf.truncated_normal([5, 5, 96, 256],     stddev=0.01), name='wc2'),
        'wc3': tf.Variable(tf.truncated_normal([3, 3, 256, 384],    stddev=0.01), name='wc3'),
        'wc4': tf.Variable(tf.truncated_normal([3, 3, 384, 384],    stddev=0.01), name='wc4'),
        'wc5': tf.Variable(tf.truncated_normal([3, 3, 384, 256],    stddev=0.01), name='wc5'),
        'wf1': tf.Variable(tf.truncated_normal([6*6*256, 4096],   stddev=0.01), name='wf1'),
        'wf2': tf.Variable(tf.truncated_normal([4096, 4096],        stddev=0.01), name='wf2'),
        'wf3': tf.Variable(tf.truncated_normal([4096, 1000],   stddev=0.01), name='wf3')
    }

    biases = {
        'bf1': tf.Variable(tf.constant(1.0, shape=[4096]), name='bf1'),
        'bf2': tf.Variable(tf.constant(1.0, shape=[4096]), name='bf2'),
        'bf3': tf.Variable(tf.constant(1.0, shape=[1000]), name='bf3')
    }

    input_layer = tf.reshape(features['x'],
                             [-1, IMAGE_SIZE, IMAGE_SIZE, 3],
                             name='data')

    # 1st conv layer
    conv1 = tf.nn.conv2d(input=input_layer,
                         filters=weights['wc1'],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='conv1')

    relu1 = tf.nn.relu(features=conv1,
                       name='relu1')

    lrn1 = tf.nn.lrn(input=relu1,
                     depth_radius=5.0,
                     bias=2.0,
                     alpha=1e-4,
                     beta=0.75,
                     name='lru1')

    max_pooling1 = tf.nn.max_pool2d(input=lrn1,
                                    ksize=3,
                                    strides=2,
                                    padding='VALID',
                                    name='max_pooling1')

    # 2nd conv layer
    conv2 = tf.nn.conv2d(input=max_pooling1,
                         filters=weights['wc2'],
                         strides=[1, 1, 1, 1],
                         padding='SAME',
                         name='conv2')

    relu2 = tf.nn.relu(features=conv2,
                       name='relu2')

    lrn2 = tf.nn.lrn(input=relu2,
                     depth_radius=5.0,
                     bias=2.0,
                     alpha=1e-4,
                     beta=0.75,
                     name='lru2')

    max_pooling2 = tf.nn.max_pool2d(input=lrn2,
                                    ksize=3,
                                    strides=2,
                                    padding='VALID',
                                    name='max_pooling2')

    # 3rd conv layer
    conv3 = tf.nn.conv2d(input=max_pooling2,
                         filters=weights['wc3'],
                         strides=[1, 1, 1, 1],
                         padding='SAME',
                         name='conv3')

    relu3 = tf.nn.relu(features=conv3,
                       name='relu3')

    # 4th conv layer
    conv4 = tf.nn.conv2d(input=relu3,
                         filters=weights['wc4'],
                         strides=[1, 1, 1, 1],
                         padding='SAME',
                         name='conv4')

    relu4 = tf.nn.relu(features=conv4,
                       name='relu4')

    # 5th conv layer
    conv5 = tf.nn.conv2d(input=relu4,
                         filters=weights['wc5'],
                         strides=[1, 1, 1, 1],
                         padding='SAME',
                         name='conv5')

    relu5 = tf.nn.relu(features=conv5,
                       name='relu5')

    max_pooling3 = tf.nn.max_pool2d(input=relu5,
                                    ksize=3,
                                    strides=2,
                                    padding='VALID',
                                    name='max_pooling3')

    # flatten 5th conv layer
    reshape = tf.reshape(max_pooling3,
                         [-1, 6 * 6 * 256],
                         name='reshape')

    # 1st fc layer
    fc1 = tf.nn.xw_plus_b(reshape,
                          weights['wf1'],
                          biases['bf1'],
                          name='fc1')

    relu6 = tf.nn.relu(features=fc1,
                       name='relu6')

    # 2nd fc layer
    fc2 = tf.nn.xw_plus_b(relu6,
                          weights['wf2'],
                          biases['bf2'],
                          name='fc2')

    relu7 = tf.nn.relu(features=fc2,
                       name='relu7')


    # 3rd fc layer, also logit layer
    fc3 = tf.nn.xw_plus_b(relu7,
                          weights['wf3'],
                          biases['bf3'],
                          name='logit')

    logits = fc3

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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

    for i in range(1, 1 + 1):
        with open(DATA_PATH + '/data_batch_' + str(i), 'rb') as f:
            train_batches = pickle.load(f, encoding='bytes')
            train_images.extend(train_batches[b'data'])
            train_labels.extend(train_batches[b'labels'])

    def preprocess_images(image):
        image.resize((32, 32, 3))
        return cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)

    train_images = [preprocess_images(image) for image in train_images]

    return labels, np.asarray(train_images), np.asarray(train_labels)


labels, train_images, train_labels = get_data()

eval_images, eval_labels = train_images[:500], train_labels[:500]
train_images, train_labels = train_images[500:], train_labels[500:]

classifier = tf.estimator.Estimator(
    model_fn=AlexNet, model_dir='./one_branch_model'
)

tensor_to_log = {'probabilities': 'softmax'}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensor_to_log, every_n_iter=50
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'x': train_images},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=False
)

classifier.train(
    input_fn=train_input_fn,
    steps=1,
    hooks=[logging_hook]
)

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
        output_node_names=['softmax']
    )
    import os
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    with tf.gfile.GFile(OUTPUT_PATH + '/frozen.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
