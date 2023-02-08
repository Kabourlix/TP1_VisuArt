import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Mise en valeur d'un objet avec un bounding box
def highlightItem(image,x1,y1,x2,y2):
    # Dessiner un carré vert
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    # Chercher taille du rectangle de text
    # Afficher texte
    cv2.rectangle(image, (x1-2, y1-70), (x2+6, y1), (0, 255, 0), -1)
    cv2.putText(image, 'Detected object num 1', (x1+20, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,0), 2)

# Code test ouvrir image d'un dossier
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
image_test = cv2.imread("data\Images\Chambre\IMG_6567.JPG") # Ajouter le chemin de l'image
imS = cv2.resize(image_test, (6000, 4000)) # sans le resize, l'affichage est trop grand
highlightItem(imS,2500,1000,3500,2100)
cv2.imshow("output", imS)
cv2.waitKey(0)

# Code test de hugo
test = np.zeros((100, 100, 3), np.uint8)
test[0:50, 0:50] = (255, 0, 0)
test[50:100, 50:100] = (0, 255, 0)
cv2.imshow("test", test)
cv2.waitKey(0)

# Convertir en grayscale + blur + threshold
cv2.namedWindow("output2", cv2.WINDOW_NORMAL)
gray_image = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
smooth_image_gb = cv2.GaussianBlur(gray_image, (15, 15), 0)
ret1, thresh1 = cv2.threshold(smooth_image_gb, 127,255, cv2.ADAPTIVE_THRESH_MEAN_C) # threshold
cv2.imshow("output2", thresh1)
cv2.waitKey(0)


def contours(filtered_image, baseImg):
    # find contours in the binary image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(baseImg, contours, -1, (0, 255, 0), 3)
    plt.figure()
    plt.imshow(baseImg)
    plt.title('Contours on the original image')
    plt.show()

contours(gray_image, imS)
