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
##############################folder level############################
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


######################################################################
############################# use contour ############################
######################################################################
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import sanghyunjo as shjo
from glob import glob
import os
import imutils

colors = shjo.get_colors()
colors[1] = [32, 167, 132]

def calculate_iou_mask(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou


mask_list = sorted(glob('./ICDAR13_FST/test/mask/*'))

for mask_path in tqdm(mask_list): # 마스크 단위

    original_mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
    mask_name = mask_path.split('/')[-1]

    gray = original_mask.copy()*255
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)

    save_mask = np.zeros(original_mask.shape, dtype=np.uint8)

    cum_mask = np.zeros(original_mask.shape, dtype=np.uint8)
    for c in cnts: # 마스크 내에 있는 박스 단위
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        c_for_box = c.reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(c_for_box)

        mask = np.zeros(original_mask.shape, dtype=np.uint8)

        left, top, right, bottom = x, y, x + w, y + h
        points = np.array([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [points], color=1)
        mask_mul = mask * original_mask
        save_mask_path = os.path.join('./ICDAR13_FST/test/transform_mask', mask_name)
        iou = calculate_iou_mask(mask, mask_mul)

        mask_mul_map = mask_mul>0
        cum_mask[(mask>0)] = 1
        if iou < 0.75:
            save_mask[mask_mul_map] = 1
        else:
            save_mask[mask_mul_map] = 255

    shjo.write_image(os.path.join('./ICDAR13_FST/test/check_mask', mask_name), cum_mask, colors)
    shjo.write_image(save_mask_path, save_mask, colors)

