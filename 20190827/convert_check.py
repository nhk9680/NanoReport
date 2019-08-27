from PIL import Image

import numpy as np

image = Image.open('2007_000032.png')
before = np.array(image)
after = np.array(image.convert('RGB'))

print("before:", before.shape)
print("after:", after.shape)

print()

print("before:", np.unique(before))
print("after:", np.unique(after))