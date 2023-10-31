#!/bin/bash
## 完整的数据集分割、训练、测试脚本

set -e

# 0. install requirements
## 安装依赖，pip安装yolov5，将会安装torch等依赖
#pip install -r requirement.txt

# 1. split the dataset. 
## 需要drone_USC.zip解压到./data/drone_USC
#mkdir -p data; unzip drone_USC.zip ./data
python main.py --task split 

# 2. train the model. 
## 训练模型，需要使用GPU环境进行，CPU训练缓慢
# 2.1 dataset description link
ln -sv ../../drone.yaml yolov5/data/drone.yaml
# 2.2 train yolo v5 model
cd yolov5
# --img: img resize to 640x640, as default
# --batch: batch size 16, as default
# --data: dataset description, new file symlink at yolov5/data/drone.yaml -> ./drone.yaml
# --weights: fine-tune from pretrained (recommended)
#python train.py --img 640 --batch 16 --epochs 100 --data drone.yaml --weights yolov5s.pt --cache
#cp runs/train/exp1/weights/best.pt ../model/yolov5s_drone.pt
## 训练后的模型已存储在./model中
# 3. test the model.
## change the weight file path as you need
python val.py --task test --data drone.yaml --weights ../model/yolov5s_drone.pt

