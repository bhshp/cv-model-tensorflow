import tensorflow as tf

if tf.__version__.startswith('2.'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
tf.disable_eager_execution()

OUTPUT_PATH='./result'

convert = tf.lite.TFLiteConverter.from_frozen_graph(OUTPUT_PATH + '/frozen.pb', input_arrays=['data'], output_arrays=['softmax'])
convert.post_training_quantize=False
tflite_model = convert.convert()

with open(OUTPUT_PATH + '/model.tflite', 'wb') as f:
    f.write(tflite_model)

convert = tf.lite.TFLiteConverter.from_frozen_graph(OUTPUT_PATH + '/frozen.pb', input_arrays=['data'], output_arrays=['softmax'])
convert.post_training_quantize=True
tflite_model = convert.convert()

with open(OUTPUT_PATH + '/model_quant.tflite', 'wb') as f:
    f.write(tflite_model)
