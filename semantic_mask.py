import cv2
import numpy as np
import sanghyunjo as shjo
import sys

# image = cv2.imread("data/KAIST_Scene_Text/test/mask/2.png")
# # print(np.unique(image))
# print(image.shape)

colors = shjo.get_colors()
colors[1] = [32, 167, 132]

input_dir = "data/HierText/val/mask_before/"
output_dir = shjo.makedir("data/HierText/val/mask/")

for mask_name in shjo.progress(shjo.listdir(input_dir)):
    mask = shjo.read_image(input_dir + mask_name)
    if mask is None:
        continue
    binary_mask = (mask > 0).astype(np.insuint8)
    mask_name = mask_name.replace("." + mask_name.split(".")[-1], ".png")
    shjo.write_image(output_dir + mask_name, binary_mask, colors)

# print(output_dir + mask_name)

# gt_mask = shjo.read_image('data/TotalText/test/mask/img1.png', mode='mask')
# print(np.unique(gt_mask))
