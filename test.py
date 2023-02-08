import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# print(f"{os.path.isfile('data/Images/Chambre/IMG_6567.JPG')} : img exists")
img = cv2.imread("data/Images/Chambre/IMG_6569.JPG")
ref = cv2.imread("data/Images/Chambre/Reference.JPG")

# Turn the images into grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)



# Plot both image
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Image to compare')
plt.subplot(122)
plt.imshow(ref, cmap='gray')
plt.title('Reference image')
plt.show()






target = cv2.subtract(img, ref)
plt.figure()
plt.imshow(target, cmap='gray')
plt.title('Difference between the two images')
plt.show()
cv2.waitKey(0)

#Apply a threshold to the target image
ret, target = cv2.threshold(target, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#Work on the target image to enhance the difference
#Dilate
kernel = np.ones((5,5),np.uint8)
target = cv2.dilate(target, kernel, iterations=1)

#Plot the target image
plt.figure()
plt.imshow(target, cmap='gray')
plt.title('Difference between the two images ENHANCED')
plt.show()