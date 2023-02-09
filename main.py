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

# Dessiner le masque à la main sur l'image : Fonction appelée seulement pour déterminer le masque
def drawMask(image):
    # cliquer sur l'image pour dessiner un mask
    listPoints = []
    def clickPoint(event, x, y, flags, params): # Quand on clique on ajoute le point à la liste et on dessine un rond
        if event == cv2.EVENT_LBUTTONDOWN:
            listPoints.append((x, y))
            cv2.circle(image, (x, y), 20, (255, 0, 255), -1)
            cv2.imshow('image', image)

    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', clickPoint) # Si appuie souris -> callback la fonction clickPoint
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(listPoints)

# Retourner masque
def getMask(image):
    mask0 = np.zeros(image.shape[:2], np.uint8)
    # Ci dessous est le masque temporaire de la chambre
    pointsMask = np.array([(2376, 3992), (2048, 3980), (2145, 3778), (2232, 3560), (2298, 3314), (2304, 3112), (2195, 2981), (2007, 2858), (1786, 2692), (1617, 2557), (1489, 2446), (1370, 2363), (1171, 2430), (1008, 2505), (852, 2585), (730, 2688), (571, 2771), (503, 2827), (456, 2696), (412, 2573), (347, 2458), (303, 2323), (268, 2208), (209, 2093), (165, 1982), (131, 1839), (84, 1744), (56, 1653), (9, 1550), (16, 1439), (3, 1265), (125, 1245), (228, 1221), (306, 1308), (387, 1411), (549, 1403), (643, 1384), (687, 1328), (777, 1332), (877, 1356), (912, 1423), (999, 1439), (1099, 1455), (1227, 1502), (1345, 1495), (1408, 1514), (1483, 1522), (1536, 1491), (1561, 1395), (1517, 1340), (1608, 1284), (1680, 1340), (1673, 1459), (1708, 1510), (1783, 1550), (1857, 1570), (2023, 1491), (2170, 1415), (2291, 1368), (2394, 1324), (2501, 1273), (2597, 1233), (2675, 1177), (2672, 1086), (2604, 1023), (2566, 944), (2504, 888), (2426, 884), (2373, 821), (2351, 781), (2338, 706), (2310, 603), (2379, 551), (2507, 547), (2560, 476), (2532, 373), (2588, 341), (2635, 440), (2647, 503), (2713, 515), (2735, 464), (2694, 369), (2750, 337), (2844, 357), (2909, 369), (2975, 373), (2988, 440), (2997, 535), (2994, 622), (2994, 694), (3000, 781), (3100, 789), (3184, 797), (3340, 844), (3471, 872), (3556, 900), (3796, 971), (4005, 1039), (4127, 1090), (4217, 1114), (4324, 1146), (4458, 1189), (4558, 1229), (4723, 1280), (4832, 1316), (4973, 1364), (5051, 1403), (5145, 1475), (5179, 1491), (5170, 1538), (5151, 1610), (5126, 1669), (5060, 1657), (4957, 1598), (4867, 1558), (4770, 1570), (4767, 1653), (4870, 1705), (4979, 1736), (5123, 1796), (5210, 1835), (5344, 1919), (5516, 2018), (5613, 2046), (5632, 2224), (5591, 2359), (5541, 2525), (5482, 2656), (5435, 2803), (5373, 2957), (5304, 3148), (5270, 3278), (5226, 3449), (5182, 3619), (5088, 3762), (5017, 3897), (4964, 3988), (4779, 3992), (4536, 3960), (4246, 3984), (3840, 3960), (3503, 3980), (3250, 3980), (2919, 3972), (2591, 3976)]);
    cv2.fillPoly(mask0, np.array([pointsMask]), 255) # remplir l'interieur de ce masque par du blanc

    # mask0[1000:3000, 1000:4000] = 255 # test

    return mask0

# Afficher masque
def showMask(image):

    masked = cv2.bitwise_and(image, image, mask=getMask(image))

    plt.figure()
    plt.subplot(121)
    plt.imshow(getMask(image))
    plt.title('Show binary mask shape')
    plt.subplot(122)
    plt.imshow(masked)
    plt.title('Show mask on image')
    plt.show()


def opening_file(path):
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

    return img


# # Code test ouvrir image d'un dossier
# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
# image_test = cv2.imread("data\Images\Chambre\IMG_6567.JPG")  # Ajouter le chemin de l'image
# imS = cv2.resize(image_test, (6000, 4000))  # sans le resize, l'affichage est trop grand
# highlightItem(imS, 2500, 1000, 3500, 2100)
# cv2.imshow("output", imS)
# cv2.waitKey(0)

# ----------------Traitement d'image----------------
# Convertir en grayscale + blur + threshold

#TODO : Modifier cette fonction ou faites en une autre pour faire le traitement d'image
def image_preprocessing_naive(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth_image_gb = cv2.GaussianBlur(gray_image, (15, 15), 0)
    ret1, thresh1 = cv2.threshold(smooth_image_gb, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C)  # threshold
    # Plot original image and the thresholded image
    plt.figure()
    plt.subplot(221)
    plt.imshow(image)
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
    return thresh1


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


if __name__ == '__main__':
    print(f"You are currently using {platform.system()}")
    print("-----------------")

    img_path = "data/Images/Chambre/IMG_6567.JPG" # TODO : Modifier le chemin de l'image ici

    img = opening_file(img_path)
    thresh = image_preprocessing_naive(img) # TODO : Utiliser votre fonction ici
    contours_detection(thresh, img)
    #drawMask(img)
    showMask(img)

