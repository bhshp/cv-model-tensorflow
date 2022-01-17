cd ./AlexNet/one_branch
rm -rf ./model ./result
clear && python train.py && python convert.py
cd ./../../
mkdir -p ./model
mv ./AlexNet/one_branch/result/model_quant.tflite ./model/AlexNet.tflite
