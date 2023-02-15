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
        sqrDist = float(pixel[0]) ** 2 + float(pixel[1]) ** 2 + float(pixel[2]) ** 2
        if sqrDist > treshold**2:
            foregroundMask[i,j] = 255

#plot the diff and the mask on the same plot
plt.subplot(121)
plt.imshow(diff)
plt.title('Difference between the two images')
plt.subplot(122)
plt.imshow(foregroundMask, cmap='gray')
plt.title('Masked difference between the two images')
plt.show()

# Dilate operation on foreground mask
kernel = np.ones((5,5),np.uint8)
dilated = cv2.dilate(foregroundMask, kernel, iterations=1)

# Opening operation on foreground mask
opening = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, kernel)
opening2 = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)

# Plot results
plt.figure()
plt.subplot(221)
plt.imshow(foregroundMask, cmap='gray')
plt.title('Masked init')
plt.subplot(222)
plt.imshow(dilated, cmap='gray')
plt.title('Dilated mask')
plt.subplot(223)
plt.imshow(opening, cmap='gray')
plt.title('Opening mask')
plt.subplot(224)
plt.imshow(opening2, cmap='gray')
plt.title('Opening mask 2')
plt.show()


#get the contours of the masked image
contours, hierarchy = cv2.findContours(foregroundMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#Get a bounding box for each contour
boundingBoxes = [cv2.boundingRect(c) for c in contours]
#Sort the bounding boxes by area
boundingBoxes = sorted(boundingBoxes, key=lambda x: x[2]*x[3], reverse=True)
#filter the bounding boxes by keeping min area
minArea = 100
boundingBoxes = [box for box in boundingBoxes if box[2]*box[3] > minArea]

#Draw the bounding boxes on the original image
for box in boundingBoxes:
    cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)

plt.figure()
plt.imshow(img)
plt.title('Contours on the original image')
plt.show()