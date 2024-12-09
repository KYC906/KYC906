from PIL import Image
import numpy as np
import cv2
import sanghyunjo as shjo

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
