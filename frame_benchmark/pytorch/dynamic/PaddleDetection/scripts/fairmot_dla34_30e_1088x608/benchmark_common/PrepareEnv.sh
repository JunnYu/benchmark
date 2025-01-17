#!/usr/bin/env bash
# 执行路径在模型库的根目录下
################################# 安装框架 如:
echo "*******prepare benchmark start ***********"
pip install -U pip
echo `pip --version`
pip install Cython pycocotools pandas
pip install torch==1.10.0 torchvision==0.11.1
pip install -r requirements.txt

################################# 准备训练数据 如:
wget -nc -P src/dataset/mot https://paddledet.bj.bcebos.com/data/mot_benchmark.tar
cd ./src/dataset/mot && tar -xf mot_benchmark.tar && mv -u mot_benchmark/* .
rm -rf mot_benchmark/ && cd ../../../
################################# 准备预训练模型 如:
wget -nc -P models/ https://paddledet.bj.bcebos.com/models/pretrained/ctdet_coco_dla_2x.pth
echo "*******prepare benchmark end***********"
