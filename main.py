import argparse
import math
import os
from os import listdir

from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
import numpy as np
import cv2

# --- Constantes ---
PX = 3  # Value to increase the area of the rectangle
DIM_IMG = (600, 400)
fig = plt.figure(figsize=(12, 6))
fig.patch.set_facecolor('silver')


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""
    # Convert BGR image to RGB
    img_rgb = color_img[:, :, ::-1]
    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')

def save_results(repo, img_name,img):
    repo = os.path.abspath(repo)
    repo = os.path.basename(repo) # Get last folder name
    if not os.path.exists('./output/' + repo):
        os.makedirs('./output/' + repo)
    cv2.imwrite('./output/' + repo + '/' + 'RESULT_' + img_name, img)

def img_load(repo):
    img_list = {}

    for im in listdir(repo):
        if im != 'Reference.JPG':
            img_list[im] = cv2.imread(repo + '/' + im)
            img_list[im] = cv2.resize(img_list[im], DIM_IMG)

    img_ref = cv2.imread(repo + '/Reference.jpg')
    img_ref = cv2.resize(img_ref, DIM_IMG)

    return img_list, img_ref


def process(img_ref, img):
    # RGB -> GREY
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ref_grey = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    abs_diff = cv2.absdiff(img_ref_grey, img_grey)

    # Gaussian Blur
    abs_diff = cv2.GaussianBlur(abs_diff, (5, 5), cv2.BORDER_DEFAULT)

    # Threshold
    abs_thresh = cv2.adaptiveThreshold(abs_diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)

    abs_thresh = cv2.erode(abs_thresh, np.ones((3, 3), np.uint8))
    abs_thresh = cv2.dilate(abs_thresh, np.ones((5, 5), np.uint8))
    return abs_thresh


def is_overlapping(bounding_box1, bounding_box2):
    """
    If one of the extremity of the first bounding box overlap one of the extremity of the second bounding box, the two bounding boxes are overlapping.
    Using the global value of PX which makes the bounding bo larger
    :param bounding_box1: The first bounding box - Rectangle [x1,y1,x2,y2] with (x1,y1) the coordinates of the left-bottom corner and (x2,y2) the coordinated of the right top corner.
    :param bounding_box2: The second bounding box - Rectangle [x1,y1,x2,y2] with (x1,y1) the coordinates of the left-bottom corner and (x2,y2) the coordinated of the right top corner.
    :return: True if the two bounding boxes are overlapping. Else otherwise
    """
    new_bounding_box1 = [bounding_box1[0] - PX, bounding_box1[1] - PX, bounding_box1[2] + PX, bounding_box1[3] + PX]
    new_bounding_box2 = [bounding_box2[0] - PX, bounding_box2[1] - PX, bounding_box2[2] + PX, bounding_box2[3] + PX]
    if (new_bounding_box2[0] >= new_bounding_box1[2]) or (new_bounding_box2[2] <= new_bounding_box1[0]) or (
            new_bounding_box2[3] <= new_bounding_box1[1]) or (new_bounding_box2[1] >= new_bounding_box1[3]):
        return False
    else:
        return True


def find_contours(thresh):
    contours = []
    edges, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in edges:
        if cv2.contourArea(contour) < 900:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        contours.append([x, y, x + w, y + h])

    for cbb in contours:
        for bb in contours:
            if cbb != bb and is_overlapping(cbb, bb):
                cbb[0] = min(cbb[0], bb[0])
                cbb[1] = min(cbb[1], bb[1])
                cbb[2] = max(cbb[2], bb[2])
                cbb[3] = max(cbb[3], bb[3])

    return cv2.groupRectangles(np.concatenate((contours, contours)), groupThreshold=1, eps=0.2)[0]

def save_image(img,repository):
    if not os.path.exists('./output' + repository):
        os.makedirs('./output' + repository)
    cv2.imwrite('./output' + repository + '/' + 'RESULT_' + img[0], img[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="The path of repository folder")

    args = parser.parse_args()

    img_list, img_ref = img_load(os.path.abspath(args.repo))

    for img_name, img in img_list.items():
        thresh = process(img_ref, img)
        contours = find_contours(thresh)

        for contour in contours:
            [x, y, w, h] = contour

            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        save_results(args.repo,img_name,img)
        show_img_with_matplotlib(img_ref, "Original Image", 1)
        show_img_with_matplotlib(img, 'RESULT_' + img_name, 2)
        # Show the Figure:
        plt.show()

        # cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(img_name, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
