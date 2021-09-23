rm -rf ./result/* ./model
python train.py
python convert.py
python test_pb.py
