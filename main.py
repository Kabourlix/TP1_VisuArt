import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import platform


# Mise en valeur d'un objet avec un bounding box
def highlightItem(image, x1, y1, x2, y2):
    # Dessiner un carré vert
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    # Chercher taille du rectangle de text
    # Afficher texte
    cv2.rectangle(image, (x1 - 2, y1 - 70), (x2 + 6, y1), (0, 255, 0), -1)
    cv2.putText(image, 'Detected object num 1', (x1 + 20, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 2)


print(f"You are currently using {platform.system()}")
print("-----------------")

# A modifier en fonction de l'image
path = "data/Images/Chambre/IMG_6567.JPG"

if platform.system() == "Windows":
    path = path.replace("/", "\\")
elif platform.system() == "Darwin":
    path = path.replace("\\", "/")

try:
    if not (os.path.isfile(path)):
        raise FileNotFoundError("Image not found")
    img = cv2.imread(path)
except FileNotFoundError as e:
    print(e)
    exit()
finally:
    print("The path exists and the image is stored in img with shape {}".format(img.shape))
# # Code test ouvrir image d'un dossier
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# image_test = cv2.imread("data\Images\Chambre\IMG_6567.JPG")  # Ajouter le chemin de l'image
# imS = cv2.resize(image_test, (6000, 4000))  # sans le resize, l'affichage est trop grand
# highlightItem(imS, 2500, 1000, 3500, 2100)
# cv2.imshow("output", imS)
# cv2.waitKey(0)

# ----------------Traitement d'image----------------
# Convertir en grayscale + blur + threshold
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
smooth_image_gb = cv2.GaussianBlur(gray_image, (15, 15), 0)
ret1, thresh1 = cv2.threshold(smooth_image_gb, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C)  # threshold
# Plot original image and the thresholded image
plt.figure()
plt.subplot(221)
plt.imshow(img)
plt.title('Original image')
plt.subplot(222)
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale image')
plt.subplot(223)
plt.imshow(smooth_image_gb, cmap='gray')
plt.title('Blurred image')
plt.subplot(224)
plt.imshow(thresh1, cmap='gray')
plt.title('Thresholded image')
plt.show()


####################

# -----------Détection et Affichage des contours----------------
def contours_detection(filtered_image, baseImg):
    # find contours in the binary image
    contours, hierarchy = cv2.findContours(filtered_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"{len(contours)} contours detected")
    cv2.drawContours(baseImg, contours, -1, (0, 255, 0), 3)
    plt.figure()
    plt.imshow(baseImg)
    plt.title('Contours on the original image')
    plt.show()


contours_detection(thresh1, img)
