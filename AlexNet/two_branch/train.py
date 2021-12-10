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
# tf.disable_eager_execution()

IMAGE_SIZE = 115
DATA_PATH = './../../data'
INPUT_CKPT_PATH = './model/model.ckpt-1'
OUTPUT_PATH = './result'
# default padding valid


def AlexNet(features, labels, mode):
    weights = {
        'wc1_branch_1': tf.Variable(tf.truncated_normal([7, 7, 3, 96],     stddev=0.01), name='wc1_branch_1'),
        'wc2_branch_1': tf.Variable(tf.truncated_normal([5, 5, 96, 256],     stddev=0.01), name='wc2_branch_1'),
        'wc3_branch_1': tf.Variable(tf.truncated_normal([3, 3, 256, 384],    stddev=0.01), name='wc3_branch_1'),
        'wc4_branch_1': tf.Variable(tf.truncated_normal([3, 3, 384, 384],    stddev=0.01), name='wc4_branch_1'),
        'wc5_branch_1': tf.Variable(tf.truncated_normal([3, 3, 384, 256],    stddev=0.01), name='wc5_branch_1'),
        'wf1_branch_1': tf.Variable(tf.truncated_normal([6*6*256, 4096],   stddev=0.01), name='wf1_branch_1'),
        'wf2_branch_1': tf.Variable(tf.truncated_normal([4096, 4096],        stddev=0.01), name='wf2_branch_1'),
        'wf3_branch_1': tf.Variable(tf.truncated_normal([4096, 1000],   stddev=0.01), name='wf3_branch_1'),

        'wc1_branch_2': tf.Variable(tf.truncated_normal([7, 7, 3, 96],     stddev=0.01), name='wc1_branch_2'),
        'wc2_branch_2': tf.Variable(tf.truncated_normal([5, 5, 96, 256],     stddev=0.01), name='wc2_branch_2'),
        'wc3_branch_2': tf.Variable(tf.truncated_normal([3, 3, 256, 384],    stddev=0.01), name='wc3_branch_2'),
        'wc4_branch_2': tf.Variable(tf.truncated_normal([3, 3, 384, 384],    stddev=0.01), name='wc4_branch_2'),
        'wc5_branch_2': tf.Variable(tf.truncated_normal([3, 3, 384, 256],    stddev=0.01), name='wc5_branch_2'),
        'wf1_branch_2': tf.Variable(tf.truncated_normal([6*6*256, 4096],   stddev=0.01), name='wf1_branch_2'),
        'wf2_branch_2': tf.Variable(tf.truncated_normal([4096, 4096],        stddev=0.01), name='wf2_branch_2'),
        'wf3_branch_2': tf.Variable(tf.truncated_normal([4096, 1000],   stddev=0.01), name='wf3_branch_2')
    }

    biases = {
        'bf1_branch_1': tf.Variable(tf.constant(1.0, shape=[4096]), name='bf1_branch_1'),
        'bf2_branch_1': tf.Variable(tf.constant(1.0, shape=[4096]), name='bf2_branch_1'),
        'bf3_branch_1': tf.Variable(tf.constant(1.0, shape=[1000]), name='bf3_branch_1'),

        'bf1_branch_2': tf.Variable(tf.constant(1.0, shape=[4096]), name='bf1_branch_2'),
        'bf2_branch_2': tf.Variable(tf.constant(1.0, shape=[4096]), name='bf2_branch_2'),
        'bf3_branch_2': tf.Variable(tf.constant(1.0, shape=[1000]), name='bf3_branch_2')
    }

    input_layer = tf.reshape(features['x'],
                             [-1, IMAGE_SIZE, IMAGE_SIZE, 3],
                             name='data')

    # 1st conv layer
    conv1_branch_1 = tf.nn.conv2d(input=input_layer,
                                  filters=weights['wc1_branch_1'],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID',
                                  name='conv1_branch_1')

    relu1_branch_1 = tf.nn.relu(features=conv1_branch_1,
                                name='relu1_branch_1')

    lrn1_branch_1 = tf.nn.lrn(input=relu1_branch_1,
                              depth_radius=5.0,
                              bias=2.0,
                              alpha=1e-4,
                              beta=0.75,
                              name='lru1_branch_1')

    max_pooling1_branch_1 = tf.nn.max_pool2d(input=lrn1_branch_1,
                                             ksize=3,
                                             strides=2,
                                             padding='VALID',
                                             name='max_pooling1_branch_1')

    # 2nd conv layer
    conv2_branch_1 = tf.nn.conv2d(input=max_pooling1_branch_1,
                                  filters=weights['wc2_branch_1'],
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name='conv2_branch_1')

    relu2_branch_1 = tf.nn.relu(features=conv2_branch_1,
                                name='relu2_branch_1')

    lrn2_branch_1 = tf.nn.lrn(input=relu2_branch_1,
                              depth_radius=5.0,
                              bias=2.0,
                              alpha=1e-4,
                              beta=0.75,
                              name='lru2_branch_1')

    max_pooling2_branch_1 = tf.nn.max_pool2d(input=lrn2_branch_1,
                                             ksize=3,
                                             strides=2,
                                             padding='VALID',
                                             name='max_pooling2_branch_1')

    # 3rd conv layer
    conv3_branch_1 = tf.nn.conv2d(input=max_pooling2_branch_1,
                                  filters=weights['wc3_branch_1'],
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name='conv3_branch_1')

    relu3_branch_1 = tf.nn.relu(features=conv3_branch_1,
                                name='relu3_branch_1')

    # 4th conv layer
    conv4_branch_1 = tf.nn.conv2d(input=relu3_branch_1,
                                  filters=weights['wc4_branch_1'],
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name='conv4_branch_1')

    relu4_branch_1 = tf.nn.relu(features=conv4_branch_1,
                                name='relu4_branch_1')

    # 5th conv layer
    conv5_branch_1 = tf.nn.conv2d(input=relu4_branch_1,
                                  filters=weights['wc5_branch_1'],
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name='conv5_branch_1')

    relu5_branch_1 = tf.nn.relu(features=conv5_branch_1,
                                name='relu5_branch_1')

    max_pooling3_branch_1 = tf.nn.max_pool2d(input=relu5_branch_1,
                                             ksize=3,
                                             strides=2,
                                             padding='VALID',
                                             name='max_pooling3_branch_1')

    # flatten 5th conv layer
    reshape_branch_1 = tf.reshape(max_pooling3_branch_1,
                                  [-1, 6 * 6 * 256],
                                  name='reshape_branch_1')

    # 1st fc layer
    fc1_branch_1 = tf.nn.xw_plus_b(reshape_branch_1,
                                   weights['wf1_branch_1'],
                                   biases['bf1_branch_1'],
                                   name='fc1_branch_1')

    relu6_branch_1 = tf.nn.relu(features=fc1_branch_1,
                                name='relu6_branch_1')

    # 2nd fc layer
    fc2_branch_1 = tf.nn.xw_plus_b(relu6_branch_1,
                                   weights['wf2_branch_1'],
                                   biases['bf2_branch_1'],
                                   name='fc2_branch_1')

    relu7_branch_1 = tf.nn.relu(features=fc2_branch_1,
                                name='relu7_branch_1')

    # 3rd fc layer, also logit layer
    fc3_branch_1 = tf.nn.xw_plus_b(relu7_branch_1,
                                   weights['wf3_branch_1'],
                                   biases['bf3_branch_1'],
                                   name='logit_branch_1')

    # ================== 2nd branch ========================

    # 1st conv layer
    conv1_branch_2 = tf.nn.conv2d(input=input_layer,
                                  filters=weights['wc1_branch_2'],
                                  strides=[1, 2, 2, 1],
                                  padding='VALID',
                                  name='conv1_branch_2')

    relu1_branch_2 = tf.nn.relu(features=conv1_branch_2,
                                name='relu1_branch_2')

    lrn1_branch_2 = tf.nn.lrn(input=relu1_branch_2,
                              depth_radius=5.0,
                              bias=2.0,
                              alpha=1e-4,
                              beta=0.75,
                              name='lru1_branch_2')

    max_pooling1_branch_2 = tf.nn.max_pool2d(input=lrn1_branch_2,
                                             ksize=3,
                                             strides=2,
                                             padding='VALID',
                                             name='max_pooling1_branch_2')

    # 2nd conv layer
    conv2_branch_2 = tf.nn.conv2d(input=max_pooling1_branch_2,
                                  filters=weights['wc2_branch_2'],
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name='conv2_branch_2')

    relu2_branch_2 = tf.nn.relu(features=conv2_branch_2,
                                name='relu2_branch_2')

    lrn2_branch_2 = tf.nn.lrn(input=relu2_branch_2,
                              depth_radius=5.0,
                              bias=2.0,
                              alpha=1e-4,
                              beta=0.75,
                              name='lru2_branch_2')

    max_pooling2_branch_2 = tf.nn.max_pool2d(input=lrn2_branch_2,
                                             ksize=3,
                                             strides=2,
                                             padding='VALID',
                                             name='max_pooling2_branch_2')

    # 3rd conv layer
    conv3_branch_2 = tf.nn.conv2d(input=max_pooling2_branch_2,
                                  filters=weights['wc3_branch_2'],
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name='conv3_branch_2')

    relu3_branch_2 = tf.nn.relu(features=conv3_branch_2,
                                name='relu3_branch_2')

    # 4th conv layer
    conv4_branch_2 = tf.nn.conv2d(input=relu3_branch_2,
                                  filters=weights['wc4_branch_2'],
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name='conv4_branch_2')

    relu4_branch_2 = tf.nn.relu(features=conv4_branch_2,
                                name='relu4_branch_2')

    # 5th conv layer
    conv5_branch_2 = tf.nn.conv2d(input=relu4_branch_2,
                                  filters=weights['wc5_branch_2'],
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name='conv5_branch_2')

    relu5_branch_2 = tf.nn.relu(features=conv5_branch_2,
                                name='relu5_branch_2')

    max_pooling3_branch_2 = tf.nn.max_pool2d(input=relu5_branch_2,
                                             ksize=3,
                                             strides=2,
                                             padding='VALID',
                                             name='max_pooling3_branch_2')

    # flatten 5th conv layer
    reshape_branch_2 = tf.reshape(max_pooling3_branch_2,
                                  [-1, 6 * 6 * 256],
                                  name='reshape_branch_2')

    # 1st fc layer
    fc1_branch_2 = tf.nn.xw_plus_b(reshape_branch_2,
                                   weights['wf1_branch_2'],
                                   biases['bf1_branch_2'],
                                   name='fc1_branch_2')

    relu6_branch_2 = tf.nn.relu(features=fc1_branch_2,
                                name='relu6_branch_2')

    # 2nd fc layer
    fc2_branch_2 = tf.nn.xw_plus_b(relu6_branch_2,
                                   weights['wf2_branch_2'],
                                   biases['bf2_branch_2'],
                                   name='fc2_branch_2')

    relu7_branch_2 = tf.nn.relu(features=fc2_branch_2,
                                name='relu7_branch_2')

    # 3rd fc layer, also logit layer
    fc3_branch_2 = tf.nn.xw_plus_b(relu7_branch_2,
                                   weights['wf3_branch_2'],
                                   biases['bf3_branch_2'],
                                   name='logit_branch_2')

    logits_branch_1 = fc3_branch_1
    logits_branch_2 = fc3_branch_2

    predictions = {
        'classes_branch_1': tf.argmax(input=logits_branch_1, axis=1),
        'probabilities_branch_1': tf.nn.softmax(logits_branch_1, name='softmax_branch_1'),
        'classes_branch_2': tf.argmax(input=logits_branch_2, axis=1),
        'probabilities_branch_2': tf.nn.softmax(logits_branch_2, name='softmax_branch_2')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    joint_loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits_branch_1) + tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits_branch_2)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=joint_loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=joint_loss, train_op=train_op)

    eval_metric_ops = {
        'accuracy_branch_1': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes_branch_1']),
        'accuracy_branch_2': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes_branch_2'])
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=joint_loss, eval_metric_ops=eval_metric_ops)


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
    model_fn=AlexNet, model_dir='./model'
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
    steps=1
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
        output_node_names=['softmax_branch_1', 'softmax_branch_2']
    )
    import os
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    with tf.gfile.GFile(OUTPUT_PATH + '/frozen.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
