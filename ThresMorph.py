# Import required packages:
import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Load the image and convert it to grayscale:
image = cv2.imread('D:\Documents\_UQAC\_Vision artificielle et traitement des images\TP1\Images\Chambre\IMG_6569.JPG')
image2 = cv2.imread('D:\Documents\_UQAC\_Vision artificielle et traitement des images\TP1\Images\Cuisine\IMG_6564.JPG')
image3 = cv2.imread('D:\Documents\_UQAC\_Vision artificielle et traitement des images\TP1\Images\Salon\IMG_6552.JPG')
image4 = cv2.imread('D:\Documents\_UQAC\_Vision artificielle et traitement des images\TP1\Images\Salon\IMG_6556.JPG')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray_image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
gray_image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)

# Apply a bilateral filter in order to reduce noise while keeping edges sharp: 
bilateral_filter = cv2.bilateralFilter(gray_image, 15, 25, 25)
bilateral_filter2 = cv2.bilateralFilter(gray_image2, 15, 25, 25)
bilateral_filter3 = cv2.bilateralFilter(gray_image3, 15, 25, 25)
bilateral_filter4 = cv2.bilateralFilter(gray_image4, 15, 25, 25)

# adaptive thresholding which chooses the best threshold value from a small region around the pixel
# (in our case, the region is a 51x51 pixel square)
# cv2.ADAPTIVE_THRESH_MEAN_C: threshold value is the mean of neighbourhood area minus C.
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C: threshold value is the weighted sum of neighbourhood values where weights are a gaussian window.
# cv2.THRESH_BINARY: if pixel value is greater than a threshold value, it is assigned one value (usually white), else it is assigned another value (usually black).
# blockSize: size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.
# C: constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.
thresh1 = cv2.adaptiveThreshold(bilateral_filter, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 5)
thresh2 = cv2.adaptiveThreshold(bilateral_filter, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
thresh3 = cv2.adaptiveThreshold(bilateral_filter2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 5)
thresh4 = cv2.adaptiveThreshold(bilateral_filter2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
thresh5 = cv2.adaptiveThreshold(bilateral_filter3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 5)
thresh6 = cv2.adaptiveThreshold(bilateral_filter3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
thresh7 = cv2.adaptiveThreshold(bilateral_filter4, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 5)
thresh8 = cv2.adaptiveThreshold(bilateral_filter4, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)

# plot the thresholded images:
show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "Adaptive thresholding - mean", 1)
show_img_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "Adaptive thresholding - gaussian", 2)
show_img_with_matplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "Adaptive thresholding - mean", 3)
show_img_with_matplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "Adaptive thresholding - gaussian", 4)
show_img_with_matplotlib(cv2.cvtColor(thresh5, cv2.COLOR_GRAY2BGR), "Adaptive thresholding - mean", 5)
show_img_with_matplotlib(cv2.cvtColor(thresh6, cv2.COLOR_GRAY2BGR), "Adaptive thresholding - gaussian", 6)
show_img_with_matplotlib(cv2.cvtColor(thresh7, cv2.COLOR_GRAY2BGR), "Adaptive thresholding - mean", 7)
show_img_with_matplotlib(cv2.cvtColor(thresh8, cv2.COLOR_GRAY2BGR), "Adaptive thresholding - gaussian", 8)

# show the Figure:
#plt.show()

# reduce the noise in the image by applying opening operation:
kernel = np.ones((15, 15), np.uint8)
opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
opening2 = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel)
opening3 = cv2.morphologyEx(thresh3, cv2.MORPH_OPEN, kernel)
opening4 = cv2.morphologyEx(thresh4, cv2.MORPH_OPEN, kernel)
opening5 = cv2.morphologyEx(thresh5, cv2.MORPH_OPEN, kernel)
opening6 = cv2.morphologyEx(thresh6, cv2.MORPH_OPEN, kernel)
opening7 = cv2.morphologyEx(thresh7, cv2.MORPH_OPEN, kernel)
opening8 = cv2.morphologyEx(thresh8, cv2.MORPH_OPEN, kernel)

# plot the opening images:
show_img_with_matplotlib(cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR), "Opening - mean", 1)
show_img_with_matplotlib(cv2.cvtColor(opening2, cv2.COLOR_GRAY2BGR), "Opening - gaussian", 2)   
show_img_with_matplotlib(cv2.cvtColor(opening3, cv2.COLOR_GRAY2BGR), "Opening - mean", 3)
show_img_with_matplotlib(cv2.cvtColor(opening4, cv2.COLOR_GRAY2BGR), "Opening - gaussian", 4)
show_img_with_matplotlib(cv2.cvtColor(opening5, cv2.COLOR_GRAY2BGR), "Opening - mean", 5)
show_img_with_matplotlib(cv2.cvtColor(opening6, cv2.COLOR_GRAY2BGR), "Opening - gaussian", 6)
show_img_with_matplotlib(cv2.cvtColor(opening7, cv2.COLOR_GRAY2BGR), "Opening - mean", 7)
show_img_with_matplotlib(cv2.cvtColor(opening8, cv2.COLOR_GRAY2BGR), "Opening - gaussian", 8)

# show the Figure:
plt.show()








