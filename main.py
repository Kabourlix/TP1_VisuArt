import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Code test ouvrir image d'un dossier
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
image_test = cv2.imread("data\Images\Chambre\IMG_6567.JPG") # Ajouter le chemin de l'image
imS = cv2.resize(image_test, (6000, 4000)) # sans le resize, l'affichage est trop grand
cv2.imshow("output", imS)
cv2.waitKey(0)

# Code test de hugo
test = np.zeros((100, 100, 3), np.uint8)
test[0:50, 0:50] = (255, 0, 0)
test[50:100, 50:100] = (0, 255, 0)
cv2.imshow("test", test)
cv2.waitKey(0)
