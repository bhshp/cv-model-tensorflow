import tensorflow as tf

print(tf.__version__)
if tf.__version__.startswith('2.'):
    tf = tf.compat.v1
    tf.disable_v2_behavior()
tf.disable_eager_execution()

OUTPUT_PATH = './two_branch_result'
OUTPUT_PB_PATH = OUTPUT_PATH + '/frozen.pb'

def convert_to_tflite():
    convert = tf.lite.TFLiteConverter.from_frozen_graph(
        OUTPUT_PB_PATH, input_arrays=['data'], output_arrays=['softmax_branch_1', 'softmax_branch_2'])
    convert.post_training_quantize = False
    tflite_model = convert.convert()

    with open(OUTPUT_PATH + '/model.tflite', 'wb') as f:
        f.write(tflite_model)

def convert_to_quant_tflite():
    convert = tf.lite.TFLiteConverter.from_frozen_graph(
        OUTPUT_PB_PATH, input_arrays=['data'], output_arrays=['softmax_branch_1', 'softmax_branch_2'])
    convert.inference_type = tf.uint8
    convert.inference_input_type = tf.uint8  # or tf.uint8
    convert.default_ranges_stats = (0,1)
    input_arrays = convert.get_input_arrays()
    print(input_arrays)
    convert.quantized_input_stats = {input_arrays[0]: (128, 1)}
    tflite_model = convert.convert()

    with open(OUTPUT_PATH + '/model_quant.tflite', 'wb') as f:
        f.write(tflite_model)

if __name__ == '__main__':
    # convert_to_tflite()
    convert_to_quant_tflite()