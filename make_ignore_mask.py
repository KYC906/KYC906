from PIL import Image
import numpy as np
import cv2
import sanghyunjo as shjo

######################################################################
##############################file level##############################
######################################################################

colors = shjo.get_colors()

mask = np.asarray(Image.open("MLT_S/train/mask/img_1.png"))
f = open("MLT_S/train/label/img_1.txt", mode='r')
box_coord = [line.strip().split(',')[:8] for line in f.readlines()]
f.close()

for coord_list in box_coord:
    coord_list = list(map(int, coord_list))
    points = np.array(coord_list, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [points], color=1)

shjo.write_image('./test.png', mask, colors)


######################################################################
##############################folder level##############################
######################################################################

import torch
import cv2
import numpy as np
from PIL import Image
import sanghyunjo as shjo
from glob import glob
import os

colors = shjo.get_colors()
colors[1] = [32, 167, 132]

def calculate_iou_mask(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou

mask_list = sorted(glob('./ICDAR13_FST/train/mask/*'))
label_list = sorted(glob('./ICDAR13_FST/train/label/*'))

for mask_path, label_path in zip(mask_list, label_list): # 마스크 단위
    original_mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
    mask_name = mask_path.split('/')[-1]
    f = open(label_path, mode='r')
    box_coord = [line.strip().split(',')[:8] for line in f.readlines()]
    f.close()

    save_mask = np.zeros(original_mask.shape, dtype=np.uint8)
    for i, coord_list in enumerate(box_coord): # 마스크 내에 있는 박스 단위
        
        mask = np.zeros(original_mask.shape, dtype=np.uint8)
        coord_list = list(map(int, coord_list))
        points = np.array(coord_list, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [points], color=1)
        mask_mul = mask * original_mask
        save_mask_path = os.path.join('./ICDAR13_FST/train/transform_mask', mask_name)
        iou = calculate_iou_mask(mask, mask_mul)
        mask_mul_map = mask_mul>0

        if iou < 0.9:
            save_mask[mask_mul_map] = 1
        else:
            save_mask[mask_mul_map] = 255

    shjo.write_image(save_mask_path, save_mask, colors)
