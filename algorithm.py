import Utility as util
import cv2
import numpy as np


def threshold(img, ref, t_value):
    diff = cv2.absdiff(img, ref)
    # TODO : Apply threshold below (naive version is below)
    thresholded = cv2.threshold(diff, t_value, 255, cv2.THRESH_BINARY)
    return thresholded


def morph_operation(img):
    # TODO : Apply morphological operation
    return img


def flood_fill(img):
    flood = img.copy()
    h, w = flood.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 0)
    return flood


def get_contours(img, init_img):
    contours = util.draw_contours_in_place(img)
    result = init_img.copy()
    util.draw_bounding_boxes_in_place(result, contours, color=(255, 0, 0), thickness=8, threshold=25000)
    return result


if __name__ == "__main__":
    path_img, path_ref = util.get_paths()
    img = util.load_img(path_img)
    ref = util.load_img(path_ref)
    # room = util.get_room(path)
    # util.quickPlot(img, f"Imported Image in {room.name}")
    util.quick_plot2(img, ref, 'Image', 'Reference', figsize=(10, 10), cmap='Greys_r', binary=True)

    th_img = threshold(img, ref, 20)
    util.quick_plot(th_img, 'Thresholded Image', figsize=(10, 10), cmap='Greys_r', binary=True)

    morph_img = morph_operation(th_img)
    util.quick_plot(morph_img, 'Morphological Operation', figsize=(10, 10), cmap='Greys_r', binary=True)

    flood_img = flood_fill(morph_img)
    util.quick_plot(flood_img, 'Flood Fill', figsize=(10, 10), cmap='Greys_r', binary=True)

    contours_img = get_contours(flood_img, img)
    util.quick_plot(contours_img, 'Contours', figsize=(10, 10), cmap='Greys_r', binary=True)
