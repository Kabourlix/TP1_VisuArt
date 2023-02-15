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
def getMask(image,path):
    mask0 = np.zeros(image.shape[:2], np.uint8)
    pointsMask = None
    room = path.split("/")[2]
    print("room : "+room)

    if(room=="Chambre"):
        # Ci dessous est le masque temporaire de la chambre
        pointsMask = np.array(
            [(2376, 3992), (2048, 3980), (2145, 3778), (2232, 3560), (2298, 3314), (2304, 3112), (2195, 2981),
             (2007, 2858), (1786, 2692), (1617, 2557), (1489, 2446), (1370, 2363), (1171, 2430), (1008, 2505),
             (852, 2585), (730, 2688), (571, 2771), (503, 2827), (456, 2696), (412, 2573), (347, 2458), (303, 2323),
             (268, 2208), (209, 2093), (165, 1982), (131, 1839), (84, 1744), (56, 1653), (9, 1550), (16, 1439),
             (3, 1265), (125, 1245), (228, 1221), (306, 1308), (387, 1411), (549, 1403), (643, 1384), (687, 1328),
             (777, 1332), (877, 1356), (912, 1423), (999, 1439), (1099, 1455), (1227, 1502), (1345, 1495), (1408, 1514),
             (1483, 1522), (1536, 1491), (1561, 1395), (1517, 1340), (1608, 1284), (1680, 1340), (1673, 1459),
             (1708, 1510), (1783, 1550), (1857, 1570), (2023, 1491), (2170, 1415), (2291, 1368), (2394, 1324),
             (2501, 1273), (2597, 1233), (2675, 1177), (2672, 1086), (2604, 1023), (2566, 944), (2504, 888),
             (2426, 884), (2373, 821), (2351, 781), (2338, 706), (2310, 603), (2379, 551), (2507, 547), (2560, 476),
             (2532, 373), (2588, 341), (2635, 440), (2647, 503), (2713, 515), (2735, 464), (2694, 369), (2750, 337),
             (2844, 357), (2909, 369), (2975, 373), (2988, 440), (2997, 535), (2994, 622), (2994, 694), (3000, 781),
             (3100, 789), (3184, 797), (3340, 844), (3471, 872), (3556, 900), (3796, 971), (4005, 1039), (4127, 1090),
             (4217, 1114), (4324, 1146), (4458, 1189), (4558, 1229), (4723, 1280), (4832, 1316), (4973, 1364),
             (5051, 1403), (5145, 1475), (5179, 1491), (5170, 1538), (5151, 1610), (5126, 1669), (5060, 1657),
             (4957, 1598), (4867, 1558), (4770, 1570), (4767, 1653), (4870, 1705), (4979, 1736), (5123, 1796),
             (5210, 1835), (5344, 1919), (5516, 2018), (5613, 2046), (5632, 2224), (5591, 2359), (5541, 2525),
             (5482, 2656), (5435, 2803), (5373, 2957), (5304, 3148), (5270, 3278), (5226, 3449), (5182, 3619),
             (5088, 3762), (5017, 3897), (4964, 3988), (4779, 3992), (4536, 3960), (4246, 3984), (3840, 3960),
             (3503, 3980), (3250, 3980), (2919, 3972), (2591, 3976)]);
    elif(room=="Cuisine"):
        # Ci dessous est le masque temporaire de la cuisine
        pointsMask = np.array(
            [(796, 3988), (815, 3941), (940, 3921), (1046, 3806), (1083, 3750), (1111, 3627), (1155, 3469),
             (1199, 3362), (1246, 3199), (1308, 3060), (1345, 2957), (1383, 2886), (1414, 2775), (1436, 2688),
             (1436, 2620), (1470, 2585), (1473, 2517), (1545, 2474), (1655, 2474), (1708, 2402), (1733, 2339),
             (1807, 2339), (1870, 2272), (1895, 2165), (1907, 2050), (1917, 1986), (2035, 1966), (2157, 1970),
             (2270, 1978), (2363, 1982), (2441, 1950), (2504, 1927), (2622, 1927), (2697, 1919), (2794, 1919),
             (2906, 1927), (2997, 1931), (3128, 1943), (3303, 1943), (3368, 1939), (3465, 1939), (3612, 1923),
             (3677, 1915), (3743, 2018), (3818, 2081), (3896, 2165), (3955, 2252), (4040, 2383), (4089, 2482),
             (4130, 2541), (4139, 2632), (4130, 2712), (4102, 2795), (4093, 2882), (4089, 2969), (4133, 3072),
             (4193, 3160), (4274, 3251), (4336, 3366), (4402, 3449), (4464, 3540), (4539, 3671), (4639, 3659),
             (4795, 3639), (4876, 3635), (4973, 3635), (5029, 3635), (5185, 3635), (5273, 3623), (5326, 3608),
             (5376, 3600), (5401, 3524), (5426, 3461), (5463, 3508), (5554, 3508), (5632, 3497), (5763, 3469),
             (5856, 3465), (5956, 3449), (5966, 3580), (5956, 3699), (5959, 3814), (5966, 3980), (5769, 3992),
             (5547, 3992), (5282, 3988), (5113, 3992), (4911, 3984), (4701, 3996), (4464, 3984), (4230, 3992),
             (4015, 3992), (3724, 3992), (3490, 3988), (3290, 3988), (3062, 3996), (2825, 3996), (2622, 3988),
             (2354, 3980), (2073, 3988), (1854, 3984), (1645, 3976), (1389, 3984), (1239, 3984), (1077, 3984),
             (937, 3992)])
    elif(room=="Salon"):
        # Ci dessous est le masque temporaire du salon
        pointsMask = np.array(
            [(6, 3976), (6, 3853), (12, 3762), (12, 3584), (9, 3366), (6, 3195), (16, 2985), (62, 3092), (91, 3207),
             (190, 3231), (240, 3195), (240, 3104), (225, 3021), (203, 2906), (184, 2779), (159, 2680), (237, 2652),
             (300, 2533), (356, 2454), (403, 2553), (437, 2605), (534, 2601), (546, 2549), (499, 2474), (475, 2426),
             (553, 2406), (631, 2398), (702, 2398), (765, 2398), (821, 2379), (880, 2355), (933, 2343), (1043, 2339),
             (1139, 2331), (1274, 2307), (1364, 2260), (1461, 2236), (1542, 2236), (1630, 2228), (1698, 2196),
             (1776, 2172), (1829, 2172), (1904, 2188), (1951, 2176), (2026, 2236), (2070, 2287), (2113, 2355),
             (2154, 2414), (2201, 2454), (2188, 2498), (2179, 2533), (2213, 2597), (2263, 2628), (2316, 2660),
             (2363, 2692), (2451, 2664), (2516, 2648), (2579, 2628), (2591, 2569), (2597, 2533), (2663, 2517),
             (2769, 2505), (2869, 2482), (2925, 2470), (3006, 2450), (3144, 2418), (3237, 2394), (3312, 2375),
             (3400, 2355), (3478, 2339), (3549, 2307), (3596, 2283), (3615, 2208), (3615, 2165), (3687, 2145),
             (3765, 2113), (3843, 2097), (3958, 2065), (4043, 2050), (4136, 2038), (4205, 2018), (4314, 2006),
             (4395, 1994), (4452, 1970), (4511, 1998), (4595, 2006), (4670, 1982), (4708, 1919), (4761, 1891),
             (4817, 1871), (4867, 1891), (4923, 1915), (4951, 1943), (4945, 2018), (4886, 2038), (4879, 2113),
             (4807, 2121), (4733, 2165), (4676, 2220), (4623, 2264), (4617, 2323), (4667, 2375), (4729, 2402),
             (4783, 2446), (4845, 2486), (4935, 2533), (4979, 2549), (5014, 2612), (5014, 2676), (5067, 2712),
             (5117, 2731), (5160, 2712), (5223, 2727), (5294, 2759), (5351, 2807), (5429, 2834), (5532, 2874),
             (5594, 2902), (5650, 2934), (5725, 2969), (5822, 3025), (5869, 3049), (5944, 3092), (5941, 3187),
             (5950, 3275), (5950, 3382), (5953, 3477), (5969, 3576), (5956, 3655), (5959, 3730), (5966, 3865),
             (5950, 3992), (5800, 3988), (5663, 3988), (5529, 3988), (5385, 3984), (5232, 3992), (5023, 3996),
             (4811, 3984), (4517, 3980), (4233, 3984), (3977, 3988), (3696, 3984), (3406, 3988), (3190, 3988),
             (2872, 3980), (2629, 3984), (2441, 3988), (2270, 3988), (2067, 3972), (1839, 3968), (1676, 3980),
             (1436, 3972), (1252, 3976), (1049, 3984), (852, 3988), (581, 3976), (446, 3976), (262, 3976), (162, 3976)])

    cv2.fillPoly(mask0, np.array([pointsMask]), 255) # remplir l'interieur de ce masque par du blanc



    return mask0

def applyMaskToImage(image,path):
    return cv2.bitwise_and(image, image, mask=getMask(image,path))

# Afficher masque
def showMask(image,path):

    masked = applyMaskToImage(image,path)

    plt.figure()
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Show binary mask shape')
    plt.subplot(132)
    plt.imshow(getMask(image,path))
    plt.title('Show binary mask shape')
    plt.subplot(133)
    plt.imshow(masked)
    plt.title('Show mask on image')
    plt.show()

    return masked


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

def getThresh(image): # même chose que preprocessing mais sans l'affichage
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth_image_gb = cv2.GaussianBlur(gray_image, (15, 15), 0)
    ret1, thresh1 = cv2.threshold(smooth_image_gb, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C)  # threshold
    return thresh1

def show_preprocessing_naive(image):
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

    return contours


if __name__ == '__main__':
    print(f"You are currently using {platform.system()}")
    print("-----------------")

    # Ouverture de l'image
    img_path = "data/Images/Cuisine/IMG_6563.JPG" # TODO : Modifier le chemin de l'image ici
    img = opening_file(img_path)

    # Afficher le preprocessing
    show_preprocessing_naive(img) # TODO : Utiliser votre fonction ici

    # Les etapes avec le mask
    # drawMask(img)
    showMask(img,img_path) # Afficher le masque
    image_with_mask = applyMaskToImage(getThresh(img),img_path) # Appliquer le masque le l'image threehold

    # Affichage contours
    contours = contours_detection(image_with_mask, img) # TODO : Mettre l'image original en 1er paramètre

    print(contours)

