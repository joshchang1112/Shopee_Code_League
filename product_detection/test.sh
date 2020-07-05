sudo pip3 install efficientnet_pytorch
mkdir reproduce_results
python3 test.py $1 models/resnet152.model reproduce_results/resnet152.csv r 152
python3 test.py $1 models/resnet152_2.model reproduce_results/resnet152_2.csv r 1522
python3 test.py $1 models/resnet101.model reproduce_results/resnet101.csv r 101
python3 test.py $1 models/densenet169.model reproduce_results/densenet169.csv d 169
python3 test.py $1 models/vgg16_bn.model reproduce_results/vgg16_bn.csv v 16
python3 test.py $1 models/efficientnet-b1.model reproduce_results/efficientnet-b1.csv e 1
python3 test.py $1 models/efficientnet-b3.model reproduce_results/efficientnet-b3.csv e 3
python3 test.py $1 models/efficientnet-b4.model reproduce_results/efficientnet-b4.csv e 4
python3 test.py $1 models/efficientnet-b5.model reproduce_results/efficientnet-b5.csv e 5
python3 test.py $1 models/efficientnet-b7.model reproduce_results/efficientnet-b7.csv e 7

