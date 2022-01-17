import tensorflow as tf

print(tf.__version__)
if tf.__version__.startswith('2.'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
tf.disable_eager_execution()

# wget http://download.tensorflow.org/models/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz
# tar -zxvf deeplabv3_xception_ade20k_train_2018_05_29.tar.gz --wildcards --no-anchored 'frozen_inference_graph.pb'

OUTPUT_PATH = './'
MODEL_FILE = './frozen.pb'

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file = MODEL_FILE, 
    input_arrays = ['sub_7'],
    output_arrays = ['ResizeBilinear_2']
)

converter.inference_type = tf.uint8
converter.inference_input_type = tf.uint8  # or tf.uint8
converter.default_ranges_stats = (0,1)
input_arrays = converter.get_input_arrays()
print(input_arrays)
converter.quantized_input_stats = {input_arrays[0]: (128, 1)}
tflite_model = converter.convert()

with open(OUTPUT_PATH + 'deeplabv3.tflite', 'wb') as f:
    f.write(tflite_model)
