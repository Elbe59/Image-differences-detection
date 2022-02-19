import argparse
import math
import os
from os import listdir

import numpy as np
import cv2
import pandas as pd
import json
import viewer
# --- Constantes ---
PX = 3  # Value to increase the area of the rectangle
RESIZE_FACTOR = 10
DIM_IMG = (600, 400)


def img_load(repo):
    img_list = {}
    repo = os.path.abspath(repo)
    repo = os.path.basename(repo) # Get last folder name
    for img in listdir('./ressources/' + repo):
        if img != 'Reference.JPG':
            img_list[img] = cv2.imread('./ressources/' + repo + '/' + img)
            img_list[img] = cv2.resize(img_list[img], DIM_IMG)

    img_ref = cv2.imread('./ressources/' + repo + '/Reference.jpg')
    img_ref = cv2.resize(img_ref, DIM_IMG)

    return img_list, img_ref


def labels_load(repo):
    repo = os.path.abspath(repo)
    repo = os.path.basename(repo) # Get last folder name
    df = pd.read_csv('./ressources/Labels/' + repo + '_labels.csv')
    labels = {}

    # labels BB
    for i in range(len(df.index) - 1):
        if not df["filename"][i] in labels:
            labels[df["filename"][i]] = []
        bb = json.loads(df["region_shape_attributes"][i].replace('""', '"'))
        labels[df["filename"][i]].append([math.floor(bb["x"] / RESIZE_FACTOR), math.floor(bb["y"] / RESIZE_FACTOR),
                                          math.floor(bb["width"] / RESIZE_FACTOR),
                                          math.floor(bb["height"] / RESIZE_FACTOR)])

    # floor coordinates
    coord = json.loads(df["region_shape_attributes"][len(df.index) - 1].replace('""', '"'))
    floor_coord = [[math.floor(coord["all_points_x"][i] / RESIZE_FACTOR),
                    math.floor(coord["all_points_y"][i] / RESIZE_FACTOR)]
                   for i in range(len(coord["all_points_x"]))]

    return labels, floor_coord


def process(img_ref, img, floor_coord):
    # RGB -> GREY
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ref_grey = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    # Diff
    abs_diff = cv2.absdiff(img_ref_grey, img_grey)

    # Gaussian Blur
    abs_diff = cv2.GaussianBlur(abs_diff, (7, 7), cv2.BORDER_DEFAULT)

    # Threshold
    abs_thresh = cv2.adaptiveThreshold(abs_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 3)

    # Floor Mask
    contour = np.array(floor_coord)
    floor_mask = np.zeros((DIM_IMG[1], DIM_IMG[0]), dtype=np.uint8)
    cv2.fillPoly(floor_mask, pts=[contour], color=(255, 255, 255))
    abs_thresh = cv2.bitwise_and(abs_thresh, abs_thresh, mask=floor_mask)

    abs_thresh = cv2.dilate(abs_thresh, np.ones((3, 3), np.uint8))
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
        if cv2.contourArea(contour) < 70:
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

    return cv2.groupRectangles(np.concatenate((contours, contours)), groupThreshold=1, eps=0.1)[0]


def get_metric(img1, img2, channel) -> float:
    # Find frequency of pixels in range between 0 and 256
    hist1 = cv2.calcHist([img1], [channel], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [channel], None, [256], [0, 256])

    # Calculate histograms and normalize it
    cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def filter_contours(img_ref, img, contours):
    filtered = []

    for contour in contours:
        [x, y, w, h] = contour

        # Crop images
        img1 = img_ref[y: h, x: w]
        img2 = img[y: h, x: w]

        # Calculate metrics for BGR
        metric_b = get_metric(img1, img2, 0)
        metric_g = get_metric(img1, img2, 1)
        metric_r = get_metric(img1, img2, 2)

        metric = (metric_b + metric_g + metric_r) / 3

        if metric < 0.95:
            filtered.append(contour)

    return filtered

def save_results(repo, img_name,img):
    repo = os.path.abspath(repo)
    repo = os.path.basename(repo) # Get last folder name
    if not os.path.exists('./output/' + repo):
        os.makedirs('./output/' + repo)
    adress = './output/' + repo + '/' + 'RESULT_' + img_name
    cv2.imwrite(adress, img)
    return adress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="The path of repository folder")

    args = parser.parse_args()

    img_list, img_ref = img_load(os.path.abspath(args.repo))
    #
    # repo = 'Salon'
    # img_list, img_ref = img_load(repo)
    labels, floor_coord = labels_load(args.repo)
    viewer.add_original_image(args.repo + "Reference.JPG")

    for img_name, img in img_list.items():
        thresh = process(img_ref, img, floor_coord)
        contours = find_contours(thresh)
        contours = filter_contours(img_ref, img, contours)

        for contour in contours:
            [x, y, w, h] = contour

            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)

        # cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
        # cv2.imshow(img_name, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        dst = save_results(args.repo,img_name,img)
        cf_matrix = [[73, 7], [7, 141]]
        data_results = [92,5,33,12]
        viewer.add_results_image(dst,img_name,cf_matrix,data_results)
    viewer.show_visualization()



if __name__ == "__main__":
    main()
