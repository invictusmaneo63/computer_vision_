from PIL import Image
import skimage as sk
from skimage import io as sk_io
import cv2
import matplotlib.pyplot as plt
import numpy as np

images = []

path_1 = "data/voc/automobile/000522.jpg"
image_1 = Image.open(path_1)
images.append(image_1)

path_2 = "data/voc/plane/000228.jpg"
image_2 = sk_io.imread(path_2)
images.append(image_2)

path_3 = "data/voc/train/000712.jpg"
image_3 = cv2.imread(path_3)
images.append(image_3)

# set up a figure of an appropriate size
fig = plt.figure(figsize=(12, 12))
num_images = len(images)

for indx in range(num_images):
    a = fig.add_subplot(1, num_images, indx+1)
    image_plot = plt.imshow(images[indx])
    a.set_title("Image " + str(indx + 1))

# plt.show()
cv_image = cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB)
plt.imshow(cv_image)
# plt.show()
print(type(cv_image))
print(type(image_2))
print(type(image_1))

pil_image = np.array(image_1)
plt.imshow(pil_image)
plt.show()
image_2 = sk.color.rgb2gray(image_2)
plt.imshow(image_2, )
# plt.show()
type(pil_image)
print(pil_image.shape)

# manipulating image
rotate_img_1 = image_1.rotate(90, expand=1)
plt.imshow(rotate_img_1)
# plt.show()

# flipping image through numpy
upended = np.flip(cv_image, axis= 0)
mirrored = np.flip(cv_image, axis= 1)

fig = plt.figure(figsize=(12,12))
a = fig.add_subplot(1, 3, 1)
plt.imshow(cv_image)
a = fig.add_subplot(1, 3, 2)
plt.imshow(upended)
a.set_title("Flipped vertically")
a = fig.add_subplot(1, 3, 3)
plt.imshow(mirrored)
a.set_title("flipped horizontally")
plt.show()
