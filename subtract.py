import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy import ndarray

import Utility as util
import glob

img = util.load_img('data/Images/Chambre/IMG_6567.JPG')
ref = util.load_img('data/Images/Chambre/Reference.JPG')


# sub = util.get_subtracted_threshold(img, ref, 20)
#
# plt.figure()
# plt.imshow(sub, cmap='Greys_r', interpolation='none')
# plt.show()
#
# refRGB = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
# #util.quick_plot2(sub, refRGB, 'Subtracted', 'Reference', figsize=(10, 10), cmap='Greys_r', binary=True)
#
#
# # double Opening
#
# kernel = np.ones((15, 15), np.uint8)
# opening = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kernel)
# opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
#
# #util.quick_plot2(sub, opening, 'Subtracted', 'Opening', figsize=(10, 10), cmap='Greys_r', binary=True)
#
# # Gaussian Blur (pas trop d'effet)
#
# blur = cv2.GaussianBlur(opening, (15, 15), 10)
#
# #util.quick_plot2(opening, blur, 'Opening', 'Gaussian Blur', figsize=(10, 10), cmap='Greys_r', binary=True)
#
#
# # FloodFill algorithm on the gaussian blur
#
# flood = blur.copy()
# h, w = flood.shape[:2]
# mask = np.zeros((h+2, w+2), np.uint8)
# cv2.floodFill(flood, mask, (0, 0), 0)
#
# #util.quick_plot2(blur, flood, 'Gaussian Blur', 'FloodFill', figsize=(10, 10), cmap='Greys_r', binary=True)
#
# # Identify contours
#
# contours = util.draw_contours_in_place(flood)
#
# print(f"Number of contours: {len(contours)}")
#
# result = img.copy()
#
# util.draw_bounding_boxes_in_place(result, contours,threshold=25000)
#
# util.quick_plot(result, 'Contours', figsize=(10, 10), cmap='Greys_r', binary=True)
#
# # Invert the floodfill
# #
# # flood_inv = cv2.bitwise_not(flood)
# #
# # util.quick_plot_any([blur, flood, flood_inv], ['Gaussian Blur', 'FloodFill', 'FloodFill Inverted'], dim = (2,2) , figsize=(10, 10), cmap='Greys_r', binary=True)
# #

def transformation(image, ref_img):
    sub = util.get_subtracted_threshold(image, ref_img, 20)
    kernel: ndarray = np.ones((15, 15), np.uint8)
    opening = cv2.morphologyEx(sub, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(opening, cv2.MORPH_OPEN, kernel)
    blur = cv2.GaussianBlur(opening, (15, 15), 10)
    flood = blur.copy()
    h, w = flood.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 0)
    contours = util.draw_contours_in_place(flood)
    result = image.copy()
    util.draw_bounding_boxes_in_place(result, contours, color=(255, 0, 0), thickness=8, threshold=25000)
    return sub, opening, blur, flood, result


if __name__ == '__main__':
    try:
        # Load all images in data/Images/Chambre, data/Images/Cuisine, data/Images/Salon except Reference.JPG and
        # store it in 3 arrays without util
        chambre = [cv2.imread(file) for file in glob.glob('data/Images/Chambre/*.JPG') if
                   file != 'data/Images/Chambre/Reference.JPG']
        cuisine = [cv2.imread(file) for file in glob.glob('data/Images/Cuisine/*.JPG') if
                   file != 'data/Images/Cuisine/Reference.JPG']
        salon = [cv2.imread(file) for file in glob.glob('data/Images/Salon/*.JPG') if
                 file != 'data/Images/Salon/Reference.JPG']

        ref = [cv2.imread('data/Images/Chambre/Reference.JPG'), cv2.imread('data/Images/Cuisine/Reference.JPG'),
               cv2.imread('data/Images/Salon/Reference.JPG')]
    except:
        print("Error while loading images")
        exit(1)

    # Apply transformation on all images
    chambre = [transformation(img, ref[0]) for img in chambre]
    cuisine = [transformation(img, ref[1]) for img in cuisine]
    salon = [transformation(img, ref[2]) for img in salon]

    # Show all subtracted images
    util.quick_plot_any([chambre[i][0] for i in range(len(chambre))], [f"Chambre Sub {i}" for i in range(len(chambre))],
                        dim=(3, 3), figsize=(10, 10), cmap='Greys_r', binary=True)
    util.quick_plot_any([cuisine[i][0] for i in range(len(cuisine))], [f"Cuisine Sub {i}" for i in range(len(cuisine))],
                        dim=(3, 3), figsize=(10, 10), cmap='Greys_r', binary=True)
    util.quick_plot_any([salon[i][0] for i in range(len(salon))], [f"Salon Sub {i}" for i in range(len(salon))],
                        dim=(4, 3), figsize=(10, 10), cmap='Greys_r', binary=True)

    # Show all opening images
    util.quick_plot_any([chambre[i][1] for i in range(len(chambre))], [f"Chambre Open {i}" for i in range(len(chambre))],
                        dim=(3, 3), figsize=(10, 10), cmap='Greys_r', binary=True)
    util.quick_plot_any([cuisine[i][1] for i in range(len(cuisine))], [f"Cuisine Open {i}" for i in range(len(cuisine))],
                        dim=(3, 3), figsize=(10, 10), cmap='Greys_r', binary=True)
    util.quick_plot_any([salon[i][1] for i in range(len(salon))], [f"Salon Open {i}" for i in range(len(salon))],
                        dim=(4, 3), figsize=(10, 10), cmap='Greys_r', binary=True)

    # Show all blur images
    util.quick_plot_any([chambre[i][2] for i in range(len(chambre))], [f"Chambre Blur {i}" for i in range(len(chambre))],
                        dim=(3, 3), figsize=(10, 10), cmap='Greys_r', binary=True)
    util.quick_plot_any([cuisine[i][2] for i in range(len(cuisine))], [f"Cuisine Blur {i}" for i in range(len(cuisine))],
                        dim=(3, 3), figsize=(10, 10), cmap='Greys_r', binary=True)
    util.quick_plot_any([salon[i][2] for i in range(len(salon))], [f"Salon Blur {i}" for i in range(len(salon))],
                        dim=(4, 3), figsize=(10, 10), cmap='Greys_r', binary=True)

    # Show all flood images
    util.quick_plot_any([chambre[i][3] for i in range(len(chambre))], [f"Chambre Flood {i}" for i in range(len(chambre))],
                        dim=(3, 3), figsize=(10, 10), cmap='Greys_r', binary=True)
    util.quick_plot_any([cuisine[i][3] for i in range(len(cuisine))], [f"Cuisine Flood {i}" for i in range(len(cuisine))],
                        dim=(3, 3), figsize=(10, 10), cmap='Greys_r', binary=True)
    util.quick_plot_any([salon[i][3] for i in range(len(salon))], [f"Salon Flood {i}" for i in range(len(salon))],
                        dim=(4, 3), figsize=(10, 10), cmap='Greys_r', binary=True)

    # Show all result images
    util.quick_plot_any([chambre[i][4] for i in range(len(chambre))], [f"Chambre Result {i}" for i in range(len(chambre))],
                        dim=(3, 3), figsize=(10, 10), cmap='Greys_r', binary=True)
    util.quick_plot_any([cuisine[i][4] for i in range(len(cuisine))], [f"Cuisine Result {i}" for i in range(len(cuisine))],
                        dim=(3, 3), figsize=(10, 10), cmap='Greys_r', binary=True)
    util.quick_plot_any([salon[i][4] for i in range(len(salon))], [f"Salon Result {i}" for i in range(len(salon))],
                        dim=(4, 3), figsize=(10, 10), cmap='Greys_r', binary=True)
