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


def get_paths():
    """
    Handle the command line arguments
    :return: The path of the image
    """
    import argparse
    #Parse to get two paths of images
    #cmd shall be : python test.py -i <path1> -r <path2>
    parser = argparse.ArgumentParser(description='Get img and a ref to detect objects')
    parser.add_argument('-i', '--img', help='The path of the image to analyze', required=True)
    parser.add_argument('-r', '--ref', help='The path of the reference image', required=True)
    args = vars(parser.parse_args())
    return args['img'], args['ref']


def quick_plot(img, title="", figsize=None, cmap="Greys", binary=True):
    """
    Plot the image
    :param binary: True if the image is binary so black and white
    :param figsize: The size of the figure
    :param title: the title of the image
    :param cmap: The color map of the image
    :param img: The image to be plotted
    """
    plt.figure()
    # Trasncript image from bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img, cmap=cmap, interpolation=None if binary else 'nearest')
    plt.title(title)
    print("The image is plotted")
    plt.show()


def quick_plot2(img1, img2, title1="", title2="", figsize=None, cmap="Greys", binary=True):
    """
    Plot the image
    :param cmap:
    :param figsize:
    :param title2:
    :param title1:
    :param img2:
    :param img1:
    :param img: The image to be plotted
    """
    plt.figure()
    plt.subplot(121)
    plt.imshow(img1, cmap=cmap, interpolation=None if binary else 'nearest')
    plt.title(title1)
    plt.subplot(122)
    plt.imshow(img2, cmap=cmap, interpolation=None if binary else 'nearest')
    plt.title(title2)
    plt.show()


def quick_plot_any(imgs, titles, dim, figsize=None, cmap="Greys", binary=True):
    """
    Plot images according to the dimension
    :param binary:
    :param cmap:
    :param figsize: The size of the figure
    :param imgs: The images to be plotted
    :param titles: The titles of the images
    :param dim: The dimension of the plot
    :return: None
    """
    if len(imgs) != len(titles):
        raise ValueError("The length of imgs and titles are not equal")
    if len(imgs) > dim[0] * dim[1]:
        raise ValueError("The length of imgs > dim are not equal")
    plt.figure(figsize=figsize)
    for i in range(len(imgs)):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(imgs[i], cmap=cmap, interpolation=None if binary else 'nearest')
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
    Draw the contours of the image (no alteration)
    :param thickness: The thickness of the contours
    :param color: The color of the contours
    :param image: The image to draw the contours from
    :return: The image with contours and the contours
    """
    img = image.copy()
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness)
    return img, contours


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
    img = image.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    return img

def highlightItem(image, x1, y1, x2, y2,index):
    # Dessiner un carré vert
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    # Chercher taille du rectangle de text
    # Afficher texte
    cv2.rectangle(image, (x1 - 2, y1 - 70), (x2 + 6, y1), (0, 255, 0), -1)
    cv2.putText(image, 'Object n°'+str(index), (x1 + 20, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 2)


def draw_bounding_boxes_in_place(image, contours, color=(0, 255, 0), thickness=4, threshold=-1):
    """
    Draw the bounding boxes of the image
    :param threshold:
    :param thickness: The thickness of the bounding box
    :param image: The image to draw the bounding boxes
    :param contours: Contours of the image
    :param color: The color of the bounding box
    :return: None
    """
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    #sort the bounding boxes by area
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[2] * x[3], reverse=True)
    # #Plot an histogram of the bounding boxes area
    # areas = [box[2] * box[3] for box in bounding_boxes]
    # plt.hist(areas, bins=20)
    # plt.show()
    if threshold != -1:
        bounding_boxes = [box for box in bounding_boxes if box[2] * box[3] > threshold]

    for i,box in enumerate(bounding_boxes):
        x, y, w, h = box
        highlightItem(image, x, y, x+w, y+h, i) # this is what is replaced by annie
        # cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness) # this is the original code
