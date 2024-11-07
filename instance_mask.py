import os
import numpy as np
import cv2
import sanghyunjo as shjo


def preprocessing_instance_mask(
    instance_mask,
):  # 인스턴스 마스크가 들어왔을 때 각 인스턴스를 gray, 0~N 까지 할당
    gray_mask = np.zeros(
        (instance_mask.shape[0], instance_mask.shape[1]), dtype=np.uint8
    )
    unique_colors = np.unique(instance_mask.reshape(-1, 3), axis=0)
    unique_colors = unique_colors[~np.all(unique_colors == [255, 255, 255], axis=1)]
    num_classes = len(unique_colors)

    for idx, color in enumerate(unique_colors):
        mask = np.all(instance_mask == color, axis=-1)
        gray_mask[mask] = idx + 4

    return gray_mask, num_classes


# colors[1] = [32, 167, 132]

input_dir = "train/mask_before/"
output_dir = shjo.makedir("train/mask/")

for mask_name in shjo.progress(shjo.listdir(input_dir)):
    instance_mask = cv2.imread(input_dir + mask_name)
    gray_mask, num_classes = preprocessing_instance_mask(instance_mask)
    colors = shjo.get_colors(num_classes)
    file_name, ext = os.path.splitext(mask_name)
    shjo.write_image(output_dir + file_name + ".png", gray_mask, colors)
