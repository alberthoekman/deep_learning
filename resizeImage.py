import cv2
from glob import glob
import os

SOURCE_IMAGES = os.path.abspath('/home/koenbuiten/Documents/deep_learning/data/sample/images')
images = glob(os.path.join(SOURCE_IMAGES, "*.png"))

# img = cv2.imread('//home/koenbuiten/Documents/deep_learning/data/sample/images/00000013_005.png', cv2.IMREAD_UNCHANGED)
# # base = os.path.basename(img)
# # Read and resize image
# base = '00000013_005.png'
# print('resizing: ',base)
# # full_size_image = cv2.imread(img)
# resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
# cv2.imwrite(os.path.join('/home/koenbuiten/Documents/deep_learning/data/sample/resizedImages',base), resized)

for img in images:
    base = os.path.basename(img)
    full_size_image = cv2.imread(img)
    resized = cv2.resize(full_size_image, (256, 256), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join('/home/koenbuiten/Documents/deep_learning/data/sample/resizedImages', base), resized)
