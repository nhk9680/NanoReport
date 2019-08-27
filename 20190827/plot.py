#http://m.blog.daum.net/geoscience/1263?categoryId=8

import cv2
from PIL import Image
import matplotlib.image as mpimg

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure() # rows*cols 행렬의 i번째 subplot 생성
rows = 2
cols = 2
 
xlabels = ["original", "OpenCV", "pillow", "matplotlib"]
filename = '2007_000032.png'
image_origin = cv2.imread(filename)

png=[0,0,0,0]

png[0] = cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB)
png[1] = image_origin
png[2] = np.array(Image.open(filename))
png[3] = mpimg.imread(filename)

for i, img in enumerate(png):
    ax = fig.add_subplot(rows, cols, i+1)
    ax.imshow(img)
    ax.set_xlabel(xlabels[i])
    ax.set_xticks([]), ax.set_yticks([])
 
plt.show()