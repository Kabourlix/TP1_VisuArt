import cv2
import matplotlib.pyplot as plt
import numpy as np

# print(f"{os.path.isfile('data/Images/Chambre/IMG_6567.JPG')} : img exists")
img = cv2.imread("data/Images/Chambre/IMG_6569.JPG")
ref = cv2.imread("data/Images/Chambre/Reference.JPG")

diff = cv2.absdiff(img, ref)
foregroundMask = np.zeros((diff.shape[0], diff.shape[1]), dtype=np.uint8)
#Apply treshold using 3 channels
dist=0
treshold = 30
for i,row in enumerate(diff):
    for j,pixel in enumerate(row):
        dist = float(pixel[0]) ** 2 + float(pixel[1]) ** 2 + float(pixel[2]) ** 2
        dist = np.sqrt(dist)
        if dist > treshold:
            foregroundMask[i,j] = 255

#plot the diff and the mask on the same plot
plt.subplot(121)
plt.imshow(diff)
plt.title('Difference between the two images')
plt.subplot(122)
plt.imshow(foregroundMask, cmap='gray')
plt.title('Masked difference between the two images')
plt.show()