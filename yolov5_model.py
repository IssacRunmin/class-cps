#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YoLo v5 model utilities
"""
import os
import random
from colorsys import hsv_to_rgb
from typing import Union, List

import numpy as np
import cv2
import torch
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.plots import Annotator as _Annotator
from yolov5.models.yolo import attempt_load, non_max_suppression


class ValidPad:
    def __init__(self, base=64) -> None:
        self.base = base
    
    def __call__(self, x:torch.Tensor) -> torch.Tensor:
        nc = x.shape[:-2]
        h, w = x.shape[-2:]
        h2 = (h+self.base-1)//self.base*self.base
        w2 = (w+self.base-1)//self.base*self.base
        x2 = torch.ones((*nc, h2, w2), device=x.device, dtype=x.dtype)
        x2[..., :h, :w] = x
        return x2

class YOLOv5:
    def __init__(self, model_dir, device, conf_thres: float = 0.25, varient: str = "m"):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        assert varient in ["n", "s", "m", "l", "x"]
        self.model = attempt_load(f"{model_dir}/yolov5{varient}.pt",
                                  device=device,
                                  fuse=True).eval()
        for k in self.model.model.children():
            if "Detect" in str(type(k)):
                k.inplace = False
        self.conf_thres = conf_thres
        self.pad = ValidPad(32)
        self.nms = lambda x: non_max_suppression(x, conf_thres=self.conf_thres)
        self.compute_loss = ComputeLoss(self.model)

    def __call__(self, x:torch.Tensor, eval=True, roi=None):
        x = self.pad(x)
        if eval:
            with torch.no_grad():
                pred = self.model(x)[0]
                ret = self.nms(pred)
            if roi is not None:
                roi = set([roi] if isinstance(roi, int) else roi)
                tmp = []
                for ret_i in ret:
                    tmp_i = []
                    for y in ret_i:
                        if int(y[-1]) in roi:
                            tmp_i.append(y)
                    if len(tmp_i):
                        tmp.append(torch.stack(tmp_i))
                    else:
                        tmp.append(torch.empty((0, 6), device=x.device))
                ret = tmp
        else:
            ret = self.model(x)[0]
        return ret

class Annotator(_Annotator):
    def __init__(self, im, line_width=None, font_size=None, font='Arial.ttf', pil=False, example='abc'):
        im = tensor2ndarray(im) if isinstance(im, torch.Tensor) else np.ascontiguousarray(im)
        super().__init__(im, line_width, font_size, font, pil, example)
        self.colorset = [
            (int(r*255), int(g*255), int(b*255)) for k in range(80) 
                for r, g, b in [hsv_to_rgb((k%20)/20, 0.5+(k//20)/8, 0.5+(k//20)/8)]
        ]
        random.seed(0)
        random.shuffle(self.colorset)
    
    def box_label(self, box, label, idx):
        color = self.colorset[idx]
        txt_color = (255, 255, 255)
        return super().box_label(box, label, color, txt_color)
    
    def box_label_all(self, pred):
        for *box, conf, cls in pred:
            cls = int(cls.item())
            conf = float(conf.item())
            label = f"{cls} {conf:.2f}"
            self.box_label(box, label, cls)

    def save(self, path):
        img = self.result()
        cv2.imwrite(path, img)

def tensor2ndarray(img: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
    """Convert torch tensor to opencv ndarray.
    1. The color space is from RGB to BGR.
    2. The pixel value is from [0., 1.] to [0, 255].
    3. Valid input: 3-dim [c, h, w] or 4-dim [b, c, h, w]
    4. Case 1: [c, h, w] or [1, c, h, w] -> [h, w, c] 
    5. Case 2: [b, c, h, w] -> [b, h, w, c] 
    """
    if isinstance(img, list):
        if len(img) == 1:
            return tensor2ndarray(img[0])
        else:
            return tensor2ndarray(torch.stack(img))
    elif isinstance(img, torch.Tensor):
        img = img.mul(255).cpu().detach()
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]
        if img.ndim == 3:
            img = img.permute(1, 2, 0)
        elif img.ndim == 4:
            img = img.permute(0, 2, 3, 1)
        else:
            raise NotImplementedError
        img2: np.ndarray = img.numpy()
        img2 = np.ascontiguousarray(img2.astype(np.uint8)[..., ::-1])
    else:
        raise NotImplementedError
    return img2