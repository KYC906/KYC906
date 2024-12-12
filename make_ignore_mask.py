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


######################################################################
############################# with canny #############################
######################################################################


from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import sanghyunjo as shjo
from glob import glob
import os
import imutils
from skimage.metrics import structural_similarity as ssim 

colors = shjo.get_colors()
colors[1] = [32, 167, 132]

def calculate_iou_mask(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou
count = 0

def compute_similarity(edge_image, contour_image):
    """
    Compute similarity between edge image and contour image.
    Args:
        edge_image (np.ndarray): Binary edge image.
        contour_image (np.ndarray): Binary contour image.
    Returns:
        float: Similarity score (e.g., Intersection over Union or Dice coefficient).
    """
    global count
    shjo.write_image(f'ttttt{count}.png', edge_image, colors)
    shjo.write_image(f'tttttt{count}.png', contour_image, colors)
    count += 1
    # Ensure both images are binary
    edge_binary = (edge_image > 0).astype(np.uint8)
    contour_binary = (contour_image > 0).astype(np.uint8)
    
    # Compute intersection and union
    intersection = np.logical_and(edge_binary, contour_binary).sum()
    union = np.logical_or(edge_binary, contour_binary).sum()
    
    # Calculate IoU (Intersection over Union)
    iou = intersection / union if union > 0 else 0.0
    
    return iou


mask_list = sorted(glob('./ICDAR13_FST/test/mask/*.png'))
image_list = sorted(glob('./ICDAR13_FST/test/image/*.png'))

if len(mask_list) != len(image_list):
    assert "not pair"

for image_path, mask_path in zip(image_list, mask_list): # 마스크 단위

    image_real = cv2.imread(image_path)
    image = cv2.imread("ICDAR13_FST/test/image/img_81.png")
    gray = cv2.cvtColor(image_real, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 40, 40) / 255
    
    original_mask = np.asarray(Image.open(mask_path), dtype=np.uint8)
    image_name = image_path.split('/')[-1]
    mask_name = mask_path.split('/')[-1]

    # mask = np.asarray(Image.open('ICDAR13_FST/test/mask/gt_img_33.png'))
    gray = original_mask.copy()*255
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    contour_image = np.zeros_like(canny)
    cv2.drawContours(contour_image, cnts[0], -1, (1), thickness=1)

    cnts = imutils.grab_contours(cnts)

    save_mask = np.zeros(original_mask.shape, dtype=np.uint8)

    cum_mask = np.zeros(original_mask.shape, dtype=np.uint8)

    canny = canny.astype(np.uint8)
    count = 0
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

        canny_remain_contour = canny * mask
        count += 1

        mask_mul = mask * original_mask
        save_mask_path = os.path.join('./ICDAR13_FST/test/transform_mask', mask_name)
        iou = calculate_iou_mask(mask, mask_mul)
        # print(np.unique(mask_mul))
        mask_mul_map = mask_mul>0
        cum_mask[(mask>0)] = 1
        if iou < 0.7:
            save_mask[mask_mul_map] = 1
        else:
            edge_iou = compute_similarity(canny_remain_contour, contour_image*mask)
            ssim_value = ssim(canny_remain_contour, contour_image*mask)
            print(ssim_value)
            save_mask[mask_mul_map] = 255
            # print(np.unique(save_mask))
    # shjo.write_image(f"canny_{image_name}.png", canny, colors)
    # shjo.write_image(f"mask_{mask_name}.png", cum_mask, colors)

    shjo.write_image(os.path.join('./ICDAR13_FST/test/check_mask', mask_name), cum_mask, colors)
    shjo.write_image(save_mask_path, save_mask, colors)
