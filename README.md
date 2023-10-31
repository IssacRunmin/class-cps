# CPS Project: Drone Detection using YoLo V5

使用YoLo v5 进行无人机检测

数据集为`drone_USC` ，datasets的文件分布如下：

```
├── README.md
├── __init__.py
├── datasets
│   ├── drone_USC				# drone_USC.zip数据集解压缩后的数据
│   │   ├── 10_10.jpg		# 图像格式为jpg
│   │   ├── 10_10.txt		# 图像label
│   │   ├── 10_100.jpg
│   │   ├── 10_100.txt
│   │   ├── ...
│   │   └── 9_990.txt
│   ├── drone_USC.zip		# 数据集
│   └── drone_USC_split	# 数据集分割后的symlink，指向drone_USC/下的文件
│       ├── test
│       ├── train
│       └── val
├── drone.yaml					# drone_USC数据集的描述，自建
├── loss.py							# loss AP at IoU
├── main.py							# 主程序
├── model								# 模型
│   └── yolov5s_drone.pt# 训练后的模型权重
├── paperwork
│   └── 无人机视觉检测课程设计题目-纠正.docx
├── run.sh							# 脚本化的本 README.md
├── utility.py					# 实用程序
└── yolov5_model.py     # 检测用的yolov5函数
```

## 完整的数据集分割、训练、测试脚本
### 0. install requirements
安装依赖，pip安装yolov5，将会安装torch等依赖
```shell
pip install yolov5 
```
### 1. split the dataset. 
需要drone_USC.zip解压到./data/drone_USC
```shell
mkdir -p data; unzip drone_USC.zip ./data
python main.py --task split 
```
### 2. train the model. 
训练模型，需要使用GPU环境进行，CPU训练缓慢
#### 2.1 dataset description link

编写数据集描述，包括train，val，test的文件夹

```shell
ln -sv ../../drone.yaml yolov5/data/drone.yaml
```
#### 2.2 train yolo v5 model

训练yolov5，项目地址：https://github.com/ultralytics/yolov5.git

```shell
#git clone https://github.com/ultralytics/yolov5.git
cd yolov5
# --img: img resize to 640x640, as default
# --batch: batch size 16, as default
# --data: dataset description, new file symlink at yolov5/data/drone.yaml -> ./drone.yaml
# --weights: fine-tune from pretrained (recommended)
python train.py --img 640 --batch 16 --epochs 100 --data drone.yaml --weights yolov5s.pt --cache
```
output:

```
train: weights=yolov5s.pt, cfg=, data=drone.yaml, hyp=data/hyps/hyp.scratch-low.yaml, epochs=100, batch_size=16, imgsz=640, rect=False, resume=False, nosave=False, noval=False, noautoanchor=False, noplots=False, evolve=None, bucket=, cache=ram, image_weights=False, device=, multi_scale=False, single_cls=False, optimizer=SGD, sync_bn=False, workers=8, project=runs/train, name=exp, exist_ok=False, quad=False, cos_lr=False, label_smoothing=0.0, patience=100, freeze=[0], save_period=-1, seed=0, local_rank=-1, entity=None, upload_dataset=False, bbox_interval=-1, artifact_alias=latest
Command 'git fetch origin' timed out after 5 seconds
YOLOv5 🚀 v7.0-228-g4d687c8 Python-3.10.13 torch-2.1.0+cu121 CUDA:0 (NVIDIA GeForce RTX 3090, 24260MiB)

hyperparameters: lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, iou_t=0.2, anchor_t=4.0, fl_gamma=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0
Comet: run 'pip install comet_ml' to automatically track and visualize YOLOv5 🚀 runs in Comet
TensorBoard: Start with 'tensorboard --logdir runs/train', view at http://localhost:6006/
Downloading https://ultralytics.com/assets/Arial.ttf to /home/runmin/.config/Ultralytics/Arial.ttf...
100%|██████████████████████████████████████████████████████████████████| 755k/755k [00:00<00:00, 820kB/s]
Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt to yolov5s.pt...
100%|███████████████████████████████████████████████████████████████| 14.1M/14.1M [00:02<00:00, 5.05MB/s]

Overriding model.yaml nc=80 with nc=1

                 from  n    params  module                                  arguments
  0                -1  1      3520  models.common.Conv                      [3, 32, 6, 2, 2]
  1                -1  1     18560  models.common.Conv                      [32, 64, 3, 2]
  2                -1  1     18816  models.common.C3                        [64, 64, 1]
  3                -1  1     73984  models.common.Conv                      [64, 128, 3, 2]
  4                -1  2    115712  models.common.C3                        [128, 128, 2]
  5                -1  1    295424  models.common.Conv                      [128, 256, 3, 2]
  6                -1  3    625152  models.common.C3                        [256, 256, 3]
  7                -1  1   1180672  models.common.Conv                      [256, 512, 3, 2]
  8                -1  1   1182720  models.common.C3                        [512, 512, 1]
  9                -1  1    656896  models.common.SPPF                      [512, 512, 5]
 10                -1  1    131584  models.common.Conv                      [512, 256, 1, 1]
 11                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 12           [-1, 6]  1         0  models.common.Concat                    [1]
 13                -1  1    361984  models.common.C3                        [512, 256, 1, False]
 14                -1  1     33024  models.common.Conv                      [256, 128, 1, 1]
 15                -1  1         0  torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']
 16           [-1, 4]  1         0  models.common.Concat                    [1]
 17                -1  1     90880  models.common.C3                        [256, 128, 1, False]
 18                -1  1    147712  models.common.Conv                      [128, 128, 3, 2]
 19          [-1, 14]  1         0  models.common.Concat                    [1]
 20                -1  1    296448  models.common.C3                        [256, 256, 1, False]
 21                -1  1    590336  models.common.Conv                      [256, 256, 3, 2]
 22          [-1, 10]  1         0  models.common.Concat                    [1]
 23                -1  1   1182720  models.common.C3                        [512, 512, 1, False]
 24      [17, 20, 23]  1     16182  models.yolo.Detect                      [1, [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]], [128, 256, 512]]
Model summary: 214 layers, 7022326 parameters, 7022326 gradients, 15.9 GFLOPs
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to runs/train/exp2
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       0/99       1.2G     0.0965    0.02309          0          7        640: 100%|██████████| 87/87 [00:38<00:00,  2.25it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 11/11 [00:04<00:00,  2.52it/s]
                   all        341        341      0.231      0.399      0.201     0.0585

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
       1/99       1.4G    0.06618     0.0185          0          9        640: 100%|██████████| 87/87 [00:24<00:00,  3.48it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 11/11 [00:03<00:00,  3.36it/s]
                   all        341        341      0.643      0.402      0.468      0.133
[..snip]
      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      98/99       1.4G    0.02539   0.006065          0          6        640: 100%|██████████| 87/87 [00:24<00:00,  3.55it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 11/11 [00:03<00:00,  3.22it/s]
                   all        341        341      0.952      0.923      0.936      0.474

      Epoch    GPU_mem   box_loss   obj_loss   cls_loss  Instances       Size
      99/99       1.4G    0.02508   0.006117          0          7        640: 100%|██████████| 87/87 [00:24<00:00,  3.60it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 11/11 [00:03<00:00,  3.63it/s]
                   all        341        341      0.952      0.924      0.935      0.472

100 epochs completed in 0.785 hours.
Optimizer stripped from runs/train/exp2/weights/last.pt, 14.3MB
Optimizer stripped from runs/train/exp2/weights/best.pt, 14.3MB

Validating runs/train/exp2/weights/best.pt...
Fusing layers...
Model summary: 157 layers, 7012822 parameters, 0 gradients, 15.8 GFLOPs
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 11/11 [00:03<00:00,  2.97it/s]
                   all        341        341      0.972      0.913      0.941      0.487
Results saved to runs/train/exp2
```

训练后的模型已存储在./model中

```shell
cp runs/train/exp1/weights/best.pt ../model/yolov5s_drone.pt
```

### 3. test the model.

评估模型对于测试集的性能

```shell
python val.py --task test --data drone.yaml --weights ../model/yolov5s_drone.pt
```

output:

```
val: data=/home/runmin/git/yolov5/data/drone.yaml, weights=['runs/train/exp2/weights/best.pt'], batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=False, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
YOLOv5 🚀 v7.0-228-g4d687c8 Python-3.10.13 torch-2.1.0+cu121 CUDA:0 (NVIDIA GeForce RTX 3090, 24260MiB)
[..snip]
                 Class     Images  Instances          P          R      mAP50   mAP50-95
                   all        341        341      0.972      0.913      0.941      0.488
Speed: 0.2ms pre-process, 1.9ms inference, 0.8ms NMS per image at shape (32, 3, 640, 640)
Results saved to runs/val/exp2
```

使用模型进行检测，输出带框的图片

```shell
python main.py --task test
```

