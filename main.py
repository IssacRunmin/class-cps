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
import torch
import torchvision
from PIL import Image

from yolov5_model import YOLOv5, Annotator
from utility import *


__author__ = 'Runmin Ou; Fanju Meng'
RANDOM_SEED = 31415
TRAIN_SPLIT_SIZE = 0.8  # 4:1
DEVICE_STR = "cpu"  # "cuda:0"
MODEL_VAR = "s_drone"  # Normal Small Median Large eXlarge. -- model varient
DATA_DIR = "./datasets/drone_USC"
MODEL_DIR = "./model"
ANNOTATION_DIR_T = "_test"
SPLIT_DIR_T = "_split"
DRONE_ROI = 0

# Argument parser and default arguments
Default_args = {
    "task": "test",
    "device" : DEVICE_STR,  # "cuda:0"
    "seed": RANDOM_SEED, 
    "dir": DATA_DIR,
    "verbose": 0
}
Device = None
Model = None
Parser = argparse.ArgumentParser("Dataset spliting and YoLo v5 model testing")
Parser.add_argument("--task", choices=("split", "test"), help="program task")
Parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
Parser.add_argument('--seed', type=int, help = "random seed")
Parser.add_argument('--dir', type=str, help = "dataset path")
Parser.add_argument('--verbose', action='store_true', help='report mAP by class')
Parser.set_defaults(**Default_args)


pil2tensor = torchvision.transforms.ToTensor()

def read_image(file_path: str):
    return pil2tensor(Image.open(file_path).convert("RGB")).unsqueeze(0)

def is_small_images(image_path, label_path, size_threshold=(32, 32)):
    # Open the image file
    with Image.open(image_path) as img:
        # Open the label file
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # Parse bounding box coordinates
                x_center, y_center, width, height = map(float, line.split()[1:])
                x1 = int((x_center - width / 2) * img.width)
                y1 = int((y_center - height / 2) * img.height)
                x2 = int((x_center + width / 2) * img.width)
                y2 = int((y_center + height / 2) * img.height)
                
                # Check if bounding box is smaller than threshold
                if (x2 - x1) < size_threshold[0] and (y2 - y1) < size_threshold[1]:
                    return True
    return False


def calculate_ap(pred_box:list, label_box:list):
    """
    calculate the AP at IoU
    """



def test(img_paths:list, out_dir:str):
    small_targets = []
    normal_targets = []
    out_subdir_path = os.path.join(out_dir, "small")
    if os.path.exists(out_subdir_path):
        shutil.rmtree(out_subdir_path)
    os.mkdir(out_subdir_path)
    out_subdir_path = os.path.join(out_dir, "normal")
    if os.path.exists(out_subdir_path):
        shutil.rmtree(out_subdir_path)
    os.mkdir(out_subdir_path)
    for img_file in img_paths: 
        label_path = img_file[:-3] + "txt"
        file_name = os.path.basename(img_file)
        dir_name = os.path.dirname(img_file)
        # img_name = '.'.join(file_name.split('.')[:-1])
        img = read_image(img_file).to(Device)
        pred = Model(img)[0].detach().cpu()
        is_small = is_small_images(img_file, label_path)
        
        # output the image with detection boxes
        ann = Annotator(img)
        ann.box_label_all(pred)
        out_subdir = "small" if is_small else "normal"
        out_path = os.path.join(out_dir, out_subdir, file_name)
        out_pred = out_path[:-3] + "txt"
        ann.save(out_path)
        # output the label
        with open(out_pred, 'w') as f:
            for *bbox, conf, cat in pred:
                if cat == DRONE_ROI:
                    b = [int(x) for x in bbox]
                    f.write(f'{cat} {b[0]} {b[1]} {b[2]} {b[3]}')
        # if is_small:
        #     small_targets.append((img, pred))
        # else:
        #     normal_targets.append((img, pred))

def split_paths(paths:list, val_ratio:float, random_seed:int):  # List[list, list]
    """
    Split image name paths (1xN) into (1-val_ratio):val_ratio
    :paths: list, image path with name strings
    :val_ratio: float, within (0,1), ratio of val
    :radom_seed: int, random seed for random.sample
    :return:  train_paths, val_paths: two image name lists
    """
    len_t = len(paths)
    val_len = int(len_t * (1-val_ratio))
    random.seed(random_seed)
    val_paths = sorted(random.sample(paths, val_len))
    train_paths = [path for path in paths if path not in set(val_paths)]
    return train_paths, val_paths

def imgs_symlink(data_path:str, out_dir:str, ori_paths:list) -> None:
    """ 
    Create symlink for images
    :data_path: string, dataset_path
    :out_dir: string, (train, val, test)
    :ori_paths: list, name list of images
    
    example:
    dataset_path=data/drone_USC
    data
    ├── drone_USC
    └── drone_USC_split
        ├── detect
        ├── train
        └── val
    """
    dataset_name = os.path.basename(data_path)
    out_path = os.path.join(data_path+SPLIT_DIR_T, out_dir)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.mkdir(out_path)
    for path in ori_paths:
        basename = os.path.basename(path)
        label_name = basename[:-3] + "txt"
        os.symlink(os.path.join("..", "..", dataset_name, basename),
                   os.path.join(out_path, basename))
        os.symlink(os.path.join("..", "..", dataset_name, label_name),
                   os.path.join(out_path, label_name))


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
        res = is_small_images(img_file, label_path)
        if res:
            small_target_imgs.append(img_file)
        else:
            normal_target_imgs.append(img_file)
    test_paths = None
    # small target
    val_paths, test_paths = split_paths(small_target_imgs, 
                                          val_ratio=TRAIN_SPLIT_SIZE, 
                                          random_seed=RANDOM_SEED)
    train_paths, val_paths = split_paths(val_paths, val_ratio=0.9, 
                                          random_seed=RANDOM_SEED)
    # imgs_symlink(dataset_path, "test_small", test_paths)
    # normal target
    val_paths_t, test_paths_t = split_paths(normal_target_imgs, 
                                              val_ratio=TRAIN_SPLIT_SIZE, 
                                              random_seed=RANDOM_SEED)
    train_paths_t, val_paths_t = split_paths(val_paths_t, val_ratio=0.9, 
                                             random_seed=RANDOM_SEED)
    # imgs_symlink(dataset_path, "test_normal", test_paths_t)
    test_paths.extend(test_paths_t)
    val_paths.extend(val_paths_t)
    train_paths.extend(train_paths_t)
    # create symlink on datasets
    out_path = dataset_path + SPLIT_DIR_T
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    imgs_symlink(dataset_path, "train", train_paths)
    imgs_symlink(dataset_path, "val", val_paths)
    imgs_symlink(dataset_path, "test", test_paths)


def test_dataset():
    global Model
    Model = YOLOv5(model_dir=MODEL_DIR, device=Device, varient=MODEL_VAR)
    # load model
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    dir_t = DATA_DIR + ANNOTATION_DIR_T
    if os.path.exists(dir_t):
        shutil.rmtree(dir_t)
    os.mkdir(dir_t)
    test_paths = list_file(os.path.join(DATA_DIR+SPLIT_DIR_T, 'test'), 'jpg')
    test(test_paths, dir_t)


if __name__ == '__main__':
    # split data
    # split_dataset(DATA_DIR)
    args = Parser.parse_args()
    DEVICE_STR = args.device
    RANDOM_SEED = args.seed
    VERBOSE = args.verbose
    DATA_DIR = args.dir
    Device = torch.device(DEVICE_STR)
    if args.task == "split":
        split_dataset(DATA_DIR)
    elif args.task == "test":
        test_dataset()