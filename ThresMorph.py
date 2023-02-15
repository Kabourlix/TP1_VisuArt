# Import required packages:
import cv2
import numpy as np
from matplotlib import pyplot as plt
import platform
import os

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

# ----------------Traitement d'image----------------

####################
# pre-traitement pour uniformiser les images

def pre_traiter_image(image,path):
    room = path.split('/')[2]
    if room == "Cuisine":
        compare_images_Cuisine(image)
    elif room == "Chambre":
        compare_images_Chambre(image)
    elif room == "Salon":
        compare_images_Salon(image)

def compare_images_Cuisine(img):
    # reference image
    ref_path = "data/Images/Cuisine/Reference.JPG"
    ref = opening_file(ref_path)
    # histogram calculation of the reference image
    hist_ref = cv2.calcHist([ref],[0],None,[256],[0,256])
    # histogram calculation of the image being treated
    hist_img = cv2.calcHist([img],[0],None,[256],[0,256])
    # compare the histograms using CHI-SQUARRED number
    ref_img_comp = cv2.compareHist(hist_ref,hist_img,cv2.HISTCMP_CHISQR)
    # print the comparison result
    print("resultat de la comparaison: ",ref_img_comp)
    if ref_img_comp >= 542154: # if condition met, modify the image
        hist_equalization(img)
    else: # if condition not met, no need to modify
        print("L'image n'a pas besoin de modification")
        return img

def compare_images_Chambre(img):
    #reference image
    ref_path = "data/Images/Chambre/Reference.JPG"
    ref = opening_file(ref_path)
    # histogram calculation for reference image
    hist_ref = cv2.calcHist([ref],[0],None,[256],[0,256])
    # histogram calculation for the image being treated
    hist_img = cv2.calcHist([img],[0],None,[256],[0,256])
    #compare the histograms
    ref_img_comp = cv2.compareHist(hist_ref,hist_img,cv2.HISTCMP_CHISQR)
    # print the comparison result
    print("resultat de la comparaison: ",ref_img_comp)
    if ref_img_comp >= 760528: # if condition met, modify the image
        hist_equalization(img)
    else: # if condition not met, no need to modify
        print("L'image n'a pas besoin de modification")
        return img

def compare_images_Salon(img):
    #reference image
    ref_path = "data/Images/Salon/Reference.JPG"
    ref = opening_file(ref_path)
    # histogram calculation of the reference image
    hist_ref = cv2.calcHist([ref],[0],None,[256],[0,256])
    # histogram calculation of the image being treated
    hist_img = cv2.calcHist([img],[0],None,[256],[0,256])
    # compare the histograms
    ref_img_comp = cv2.compareHist(hist_ref,hist_img,cv2.HISTCMP_CHISQR)
    # print the result of the comparison
    print("resultat de la comparaison: ",ref_img_comp)
    if ref_img_comp >= 2067559: # if condition is met, modify the image
       adjust_light_salon(img)
    else: # if condition not met, no need to modify
        print("L'image n'a pas besoin de modification")
        return img

def adjust_light_salon(img):
    # convert the image to grayscale
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ref_path = "data/Images/Salon/Reference.JPG"
    # reference image
    ref = opening_file(ref_path)
    ref_gray = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
    # Convert to HLS
    hls_Img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # decreasing the L channel by a factor from the original
    hls_Img[..., 1] = hls_Img[..., 1] * 0.8
    # the HLS converted image
    img_hls=cv2.cvtColor(hls_Img, cv2.COLOR_HLS2BGR)
    # grayscale HLS image
    img_hls_gray = cv2.cvtColor(img_hls, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(10,6))
    # show the image being treated in grayscale
    plt.subplot(131)
    plt.imshow(gray_img, cmap='gray')
    plt.title('gray image')
    # show the reference image in grayscale
    plt.subplot(132)
    plt.imshow(ref_gray, cmap='gray')
    plt.title('gray image reference')
    # show the treated image in grayscale
    plt.subplot(133)
    plt.imshow(img_hls_gray, cmap='gray')
    plt.title('brightness reduced image')
    plt.show()
    return img_hls_gray


def hist_equalization(img1):
    # convert the image being treated in grayscale
    gray_original = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # histogram calculation of the original image
    hist_gray_original = cv2.calcHist([gray_original], [0], None, [256], [0, 256])
    # Create a matrix to be added to the image for saturation increase
    M = np.ones(img1.shape, dtype="uint8")*85
    # the matrix is added to the image
    added_img = cv2.add(img1,M)
    # convert the resulting image to grayscale
    gray_image = cv2.cvtColor(added_img, cv2.COLOR_BGR2GRAY)
    # gray image histogram calculation
    hist_gray = cv2.calcHist([gray_image],[0], None, [256],[0,256])
    # CLAHE applied to the gray_img
    clahe = cv2.createCLAHE(clipLimit = 4.0)
    gray_img_clahe = clahe.apply(gray_image)
    # histogram calculation of the Clahe image
    hist_gray_clahe = cv2.calcHist([gray_img_clahe],[0],None,[256],[0,256])
    # show the histograms
    plt.figure(figsize=(10,16))
    plt.subplot(321)
    plt.imshow(gray_original, cmap='gray')
    plt.title('Original gray image')
    plt.subplot(322)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist_gray_original, color='m')
    plt.title('Grayscale image histogram')
    plt.subplot(323)
    plt.imshow(gray_image, cmap='gray')
    plt.title('gray image saturee')
    plt.subplot(324)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist_gray, color='m')
    plt.title('Grayscale image histogram')
    plt.subplot(325)
    plt.imshow(gray_img_clahe, cmap='gray')
    plt.title('gray image saturee clahe')
    plt.subplot(326)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.title('Gray clahe histogram')
    plt.plot(hist_gray_clahe, color='m')
    plt.show()
    return gray_img_clahe

####################
# Convertir en grayscale + blur + threshold

def show_preprocessing_naive(img_hls_gray):
    smooth_image_gb = cv2.GaussianBlur(img_hls_gray, (15, 15), 0)
    ret1, thresh1 = cv2.threshold(smooth_image_gb, 127, 255, cv2.ADAPTIVE_THRESH_MEAN_C)  # threshold
    # Plot original image and the thresholded image
    plt.figure()
    plt.subplot(221)
    plt.imshow(img_hls_gray)
    plt.title('Original image')
    plt.subplot(222)
    plt.imshow(smooth_image_gb, cmap='gray')
    plt.title('Blurred image')
    plt.subplot(223)
    plt.imshow(thresh1, cmap='gray')
    plt.title('Thresholded image')
    plt.show()
    return thresh1

####################
# Opérations morphologiques

def show_morphological_operation(thresh1):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(thresh1, kernel, iterations=1)
    dilation = cv2.dilate(thresh1, kernel, iterations=1)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    plt.figure()
    plt.subplot(221)
    plt.imshow(closing, cmap='gray')
    plt.title('Closing image')
    plt.subplot(222)
    plt.imshow(erosion, cmap='gray')
    plt.title('Erosion image')
    plt.subplot(223)
    plt.imshow(dilation, cmap='gray')
    plt.title('Dilation image')
    plt.subplot(224)
    plt.imshow(opening, cmap='gray')
    plt.title('Opening image')
    plt.show()
    return opening

####################

if __name__ == '__main__':
    print(f"You are currently using {platform.system()}")
    print("-----------------")

    # Ouverture de l'image
    img_path = "data/Images/Cuisine/IMG_6563.JPG" # TODO : Modifier le chemin de l'image ici
    img_path1 = "data/Images/Cuisine/IMG_6565.JPG"
    img = opening_file(img_path)
    img1 = opening_file(img_path1)
    pre_traiter_image(img, img_path)
    pre_traiter_image(img1,img_path1)

    # Afficher le preprocessing
    t = show_preprocessing_naive(img) # TODO : Utiliser votre fonction ici

    # Afficher les opérations morphologiques
    show_morphological_operation(t) # TODO : Utiliser votre fonction ici







