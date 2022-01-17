cd ./Inception\ V3/two_branch
rm -rf ./model ./result
clear && python train.py && python convert.py
cd ./../../
mkdir -p ./model
mv ./Inception\ V3/two_branch/result/model_quant.tflite ./model/InceptionV3.tflite
