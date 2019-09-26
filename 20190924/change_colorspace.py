import cv2
import numpy as np

img = cv2.imread("D:\datasets\ThermalWorld_VOC_v1_0\dataset\train\SegmentationObject\IMG_0009.png")
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
cv2.imshow(img)