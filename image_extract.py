import os
from glob import glob
import cv2
from tqdm import tqdm

root_dir_mask = "data/MLT_S/train/mask"
root_dir_image = "data/MLT_S/train/image_before"

mask_paths = glob(os.path.join(root_dir_mask, "*.png"))
image_paths = glob(os.path.join(root_dir_image, "*"))

image_list = []

mask_names = []

for i in mask_paths:
    mask_names.append(i.split("/")[-1].split(".")[0])
print(len(image_paths))
for image_path in tqdm(image_paths):
    image = cv2.imread(image_path)
    if image is None:
        image = cv2.VideoCapture(image_path)
        ret, image = image.read()
    image_name = image_path.split("/")[-1]
    image_name = image_name.split(".")[0]
    if image_name in mask_names:
        image_list.append(image_path)
        mask_names.remove(image_name)
        cv2.imwrite(os.path.join("data/MLT_S/train/image", image_name + ".png"), image)

print(len(image_list))
print(mask_names)
