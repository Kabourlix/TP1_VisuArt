import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import platform
from enum import Enum


class Room(Enum):
    Unknown = 0
    Kitchen = 1
    Chamber = 2
    LivingRoom = 3


def load_img(path):
    """
    Open the image and return the image
    :param path: The relative path of the image
    :return: The image to be used
    """
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


def get_room(path):
    """
    Get the room name from the path
    :param path: The relative path of the image
    :return: The room via enum
    """
    if platform.system() == "Windows":
        path = path.replace("/", "\\")
    elif platform.system() == "Darwin":
        path = path.replace("\\", "/")
    room = Room.Unknown
    room_name = "Unknown"
    try:
        if not (os.path.isdir(path)):
            raise FileNotFoundError("Folder not found")
        room_name = path.split("/")[-1]
        if room_name == "Chamber":
            room = Room.Chamber
        elif room_name == "Kitchen":
            room = Room.Kitchen
        elif room_name == "LivingRoom":
            room = Room.LivingRoom
    except FileNotFoundError as e:
        print(e)
        exit()
    finally:
        print("The path exists and the room is {}".format(room_name))
    return room


def get_path():
    """
    Handle the command line arguments
    :return: The path of the image
    """
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('path', metavar='path', type=str, nargs=1,
                        help='the path of the image')
    args = parser.parse_args()
    return args.path[0]


def quickPlot(img, title="", figsize=None, cmap="Greys"):
    """
    Plot the image
    :param figsize: The size of the figure
    :param title: the title of the image
    :param cmap: The color map of the image
    :param img: The image to be plotted
    """
    plt.figure()
    #Trasncript image from bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(title)
    print("The image is plotted")
    plt.show()


def quick_plot2(img1, img2, title1="", title2="", figsize=None):
    """
    Plot the image
    :param img: The image to be plotted
    """
    plt.figure()
    plt.subplot(121)
    plt.imshow(img1)
    plt.title(title1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.title(title2)
    plt.show()


def quick_plot_any(imgs, titles, dim, figsize=None):
    """
    Plot images according to the dimension
    :param figsize: The size of the figure
    :param imgs: The images to be plotted
    :param titles: The titles of the images
    :param dim: The dimension of the plot
    :return: None
    """
    if len(imgs) != len(titles):
        raise ValueError("The length of imgs and titles are not equal")
    if len(imgs) != dim[0] * dim[1]:
        raise ValueError("The length of imgs and dim are not equal")
    plt.figure(figsize=figsize)
    for i in range(len(imgs)):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(imgs[i])
        plt.title(titles[i])
    plt.show()


def get_subtracted_threshold_naive(img, ref, threshold, t_param=cv2.THRESH_BINARY):
    """
    Get the thresholded image
    :param img: The image to be thresholded
    :param ref: The reference image
    :param threshold: The threshold
    :return: The thresholded image
    """
    diff = cv2.absdiff(img, ref)
    _, thresholded = cv2.threshold(diff, threshold, 255, t_param)
    return thresholded


def get_subtracted_threshold(img, ref, threshold):
    """
    Get the thresholded image using "multi channel thresholding"
    :param img: The image to be thresholded
    :param ref: The reference image
    :param threshold: The threshold
    :return: The thresholded image (foreground mask)
    """
    diff = cv2.absdiff(img, ref)
    return ((diff ** 2).sum(axis=2) > threshold).astype(np.uint8) * 255


def draw_contours(image, color=(0, 255, 0), thickness=2):
    """
    Draw the contours of the image (alteration)
    :param thickness: The thickness of the contours
    :param color: The color of the contours
    :param image: The image to draw the contours from
    :return: The image with contours and the contours
    """
    img = image.deepcopy()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(img, [contour], -1, color, thickness)
    return image, contours


def draw_contours_in_place(image, color=(0, 255, 0), thickness=2):
    """
    Draw the contours of the image (alteration)
    :param thickness: The thickness of the contours
    :param color: The color of the contours
    :param image: The image to draw the contours on
    :return: the contours
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(image, [contour], -1, color, thickness)
    return contours


def draw_bounding_boxes(image, contours, color=(0, 255, 0), thickness=2):
    """
    Draw the bounding boxes of the image (no alteration)
    :param thickness: The thickness of the bounding box
    :param color: The color of the bounding box
    :param contours: Contours of the image
    :param image: The image to draw the bounding boxes from
    :return: The image with bounding boxes
    """
    img = image.deepcopy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    return img


def draw_bounding_boxes_in_place(image, contours, color=(0, 255, 0), thickness=2):
    """
    Draw the bounding boxes of the image
    :param thickness: The thickness of the bounding box
    :param image: The image to draw the bounding boxes
    :param contours: Contours of the image
    :param color: The color of the bounding box
    :return: None
    """
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
