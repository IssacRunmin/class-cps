#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Drone object detection demo
"""
import os
import random
import shutil
import argparse

# import numpy as np
# import torch
# import torchvision
# from PIL import Image

# from yolov5_model import YOLOv5, Annotator
from utility import *


__author__ = 'Runmin Ou; Fanju Meng'
RANDOM_SEED = 31415
TRAIN_SPLIT_SIZE = 0.8  # 4:1
DEVICE_STR = "cpu"  # "cuda:0"
MODEL_VAR = "s"  # Normal Small Median Large eXlarge. -- model varient
DATA_DIR = "./data/drone_USC"
MODEL_DIR = "./model"
ANNOTATION_DIR_T = "_annotated_img"

Device = None
Model = None

# pil2tensor = torchvision.transforms.ToTensor()

# def read_image(file_path: str):
#     return pil2tensor(Image.open(file_path).convert("RGB")).unsqueeze(0)


def validate(img_paths:list):
    for img_file in img_paths: 
        label_path = img_file[:-3] + "txt"
        file_name = os.path.basename(img_file)
        dir_name = os.path.dirname(img_file)
        # img_name = '.'.join(file_name.split('.')[:-1])
        img = read_image(img_file).to(Device)
        pred = Model(img)[0].detach().cpu()
        # output the image with detection boxes
        ann = Annotator(img)
        ann.box_label_all(pred)
        out_path = os.path.join(dir_name + ANNOTATION_DIR_T, file_name)
        ann.save(out_path)


def split_dataset(dataset_path: str) -> None:
    """
    split the dataset, detect the small target image and then split then into different set
    train: training data without small targets, 80%
    detect: detection data without small targets, 20%
    """
    img_paths = list_file(dataset_path, 'jpg')
    data_name = os.path.basename(dataset_path)
    data_path = os.path.dirname(dataset_path)
    small_target_imgs = []
    normal_target_imgs = []
    for img_file in img_paths:
        label_path = img_file[:-3] + "txt"
        res = extract_small_images(img_file, label_path)
        if res:
            small_target_imgs.append(img_file)
        else:
            normal_target_imgs.append(img_file)
    detect_paths = None
    # small target
    imgs = sorted(small_target_imgs)
    dataset_len = len(imgs)
    val_len = int(dataset_len * (1-TRAIN_SPLIT_SIZE))
    random.seed(RANDOM_SEED)
    val_paths = sorted(random.sample(imgs, val_len))
    detect_paths = val_paths
    train_paths = [path for path in imgs if path not in set(val_paths)]
    # normal target
    imgs = sorted(normal_target_imgs)
    dataset_len = len(imgs)
    val_len = int(dataset_len * (1-TRAIN_SPLIT_SIZE))
    random.seed(RANDOM_SEED)
    val_paths = sorted(random.sample(imgs, val_len))
    detect_paths.extend(val_paths)
    train_paths.extend([path for path in imgs if path not in set(val_paths)])
    # create symlink on datasets
    # train data
    dir_t = os.path.join(data_path, 'train')
    if os.path.exists(dir_t):
        shutil.rmtree(dir_t)
    os.mkdir(dir_t)
    for path in train_paths:
        basename = os.path.basename(path)
        label_name = basename[:-3] + "txt"
        os.symlink(os.path.join("..", data_name, basename),
                   os.path.join(dir_t, basename))
        os.symlink(os.path.join("..", data_name, label_name),
                   os.path.join(dir_t, label_name))
    # detect data
    dir_t = os.path.join(data_path, 'detect')
    if os.path.exists(dir_t):
        shutil.rmtree(dir_t)
    os.mkdir(dir_t)
    for path in detect_paths:
        basename = os.path.basename(path)
        label_name = basename[:-3] + "txt"
        os.symlink(os.path.join("..", data_name, basename),
                   os.path.join(dir_t, basename))
        os.symlink(os.path.join("..", data_name, label_name),
                   os.path.join(dir_t, label_name))


def main():
    # load model
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    dir_t = DATA_DIR + ANNOTATION_DIR_T
    if os.path.exists(dir_t):
        shutil.rmtree(dir_t)
    os.mkdir(dir_t)
    # split data
    split_dataset(DATA_DIR)
    # img_paths = list_file(DATA_DIR, "jpg")
    # dataset_len = len(img_paths)
    # # validate model
    # val_len = int(dataset_len * (1-TRAIN_SPLIT_SIZE))
    # val_paths = sorted(random.sample(img_paths, val_len))
    # validate(val_paths)


if __name__ == '__main__':
    # Device = torch.device(DEVICE_STR)
    # Model = YOLOv5(model_dir=MODEL_DIR, device=Device, varient=MODEL_VAR)
    main()