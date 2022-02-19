import os
from os import listdir
import json
import math
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import structural_similarity
import torch
import torchvision.ops.boxes as bops
import viewer
import argparse


# --- Constantes ---
CHAMBRE_REPO = 'Chambre'
CUISINE_REPO = 'Cuisine'
SALON_REPO = 'Salon'
DIM_IMG = (600, 400)
RESIZE_FACTOR = 10
SEUIL_BB_SIZE = 400
SEUIL_IOU = 0.5
SEUIL_CROSSCORR = 0.8
PX = 0


def img_load(repo):
    """
    Description
    :param
    :return:
    """

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
    """
    Description
    :param
    :return:
    """
    repo = os.path.abspath(repo)
    repo = os.path.basename(repo) # Get last folder name
    df = pd.read_csv('./ressources/Labels/' + repo + '_labels.csv')
    labels = {}

    # labels BB
    for i in range(len(df.index) - 1):
        if not df["filename"][i] in labels:
            labels[df["filename"][i]] = []
        bb = json.loads(df["region_shape_attributes"][i].replace('""', '"'))
        labels[df["filename"][i]].append([resize(bb["x"]), resize(bb["y"]), resize(bb["width"]), resize(bb["height"])])

    # floor coordinates
    coord = json.loads(df["region_shape_attributes"][len(df.index) - 1].replace('""', '"'))
    floor_coord = [[resize(coord["all_points_x"][i]), resize(coord["all_points_y"][i])]
                   for i in range(len(coord["all_points_x"]))]

    return labels, floor_coord


def resize(coord):
    """
    Description
    :param
    :return:
    """

    return math.floor(coord / RESIZE_FACTOR)


def save_results(repo, img_name, img):
    """
    Description
    :param
    :return:
    """

    repo = os.path.abspath(repo)
    repo = os.path.basename(repo) # Get last folder name
    if not os.path.exists('./results/' + repo):
        os.makedirs('./results/' + repo)
    path = './results/' + repo + '/' + 'RESULT_' + img_name
    cv2.imwrite(path, img)
    return path



def is_overlapping(bb1, bb2, seuil):
    """
    Description
    :param
    :return:
    """
    if bops.box_iou(torch.tensor([bb1]), torch.tensor([bb2])).item() > seuil:
        return True
    else:
        return False


def sort_overlapping_bb(bb_array):
    """
    Description
    :param
    :return:
    """

    sorted_bb_array = []

    for cbb in bb_array:
        for bb in bb_array:
            if cbb != bb and is_overlapping(cbb, bb, 0.05):
                cbb[0] = min(cbb[0], bb[0])
                cbb[1] = min(cbb[1], bb[1])
                cbb[2] = max(cbb[2], bb[2])
                cbb[3] = max(cbb[3], bb[3])


    for cbb in bb_array:
        if cbb not in sorted_bb_array:
            sorted_bb_array.append(cbb)

    return sorted_bb_array


def confusion_matrix(bb, lb):
    """
    Description
    :param
    :return:
    """

    bb_count = [False] * len(bb)
    lb_count = [False] * len(lb)

    for i in range(len(bb)):
        for j in range(len(lb)):
            if is_overlapping(bb[i], lb[j], SEUIL_IOU):
                bb_count[i] = True
                lb_count[i] = True

    nb_predict = len(bb)
    tp = sum(bb_count)
    fp = len(bb_count) - sum(bb_count)
    fn = len(lb_count) - sum(lb_count)

    return tp, fp, fn, nb_predict


def calc_metrics(tp, fp, fn, nb_predict):
    """
    Description
    :param
    :return:
    """

    accuracy = tp / nb_predict
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = (2 * precision * recall) / (precision + recall)
    return accuracy, recall, precision, f1_score


def process(img, img_ref, floor_coord):
    """
    Description
    :param
    :return:
    """

    # RGB -> GREY
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ref_grey = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    # diff
    struct_diff = structural_similarity(img_ref_grey, img_grey, full=True)[1]
    struct_diff = (struct_diff * 255).astype("uint8")

    abs_diff = cv2.absdiff(img_ref_grey, img_grey)

    # apply a bilateral filter in order to reduce noise while keeping the edges sharp:
    struct_diff = cv2.bilateralFilter(struct_diff, 10, 50, 50)
    abs_diff = cv2.bilateralFilter(abs_diff, 25, 80, 80)

    # threshold
    struct_thresh = cv2.threshold(struct_diff, 80, 255, cv2.THRESH_BINARY_INV)[1]
    abs_thresh = cv2.threshold(abs_diff, 20, 255, cv2.THRESH_BINARY)[1]

    # erode/dilate
    struct_thresh = cv2.dilate(struct_thresh, np.ones((3, 3), np.uint8))
    struct_thresh = cv2.erode(struct_thresh, np.ones((3, 3), np.uint8))

    # bitwise and
    final_thresh = np.bitwise_and(struct_thresh, abs_thresh)

    # floor mask
    contours = np.array(floor_coord)
    floor_mask = np.zeros((DIM_IMG[1], DIM_IMG[0]), dtype=np.uint8)
    cv2.fillPoly(floor_mask, pts=[contours], color=(255, 255, 255))
    final_thresh = cv2.bitwise_and(final_thresh, final_thresh, mask=floor_mask)

    # display
    """
    cv2.imshow("test", final_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    return final_thresh


def find_contours(thresh):
    """
    Description
    :param
    :return:
    """

    contours = []
    edges, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    for contour in edges:
        if cv2.contourArea(contour) < SEUIL_BB_SIZE:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)

        contours.append([x, y, x + w, y + h])

    for i in range(2):
        sort_overlapping_bb(contours)

    return contours


def cross_corr_hist(img1, img2, channel) -> float:
    """
    Description
    :param
    :return:
    """

    # Find frequency of pixels in range between 0 and 256
    hist1 = cv2.calcHist([img1], [channel], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [channel], None, [256], [0, 256])

    # Calculate histograms and normalize it
    cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def filter_contours(img_ref, img, contours):
    """
    Description
    :param
    :return:
    """

    filtered = []

    for contour in contours:
        [x1, y1, x2, y2] = contour

        # Crop images
        img1 = img_ref[x1: x2, y1: y2]
        img2 = img[x1: x2, y1: y2]

        # Calculate metrics for BGR
        cch_b = cross_corr_hist(img1, img2, 0)
        cch_g = cross_corr_hist(img1, img2, 1)
        cch_r = cross_corr_hist(img1, img2, 2)

        cch_total = (cch_b + cch_g + cch_r) / 3

        if cch_total < SEUIL_CROSSCORR:
            filtered.append(contours)

    return filtered


def draw_bb(img, bb_array): # à modifier
    """
    Description
    :param
    :return:
    """

    for bb in bb_array:
        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 3)

    return img


def main():
    """
    Description
    :param
    :return:
    """


    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="The path of repository folder")
    args = parser.parse_args()

    img_list, img_ref = img_load(os.path.abspath(args.repo))

    # repo = 'Cuisine'
    # img_list, img_ref = img_load(repo)
    # labels, floor_coord = labels_load(repo)
    labels, floor_coord = labels_load(args.repo)
    viewer.add_original_image(args.repo + "Reference.JPG")

    for img_name, img in img_list.items():
        thresh = process(img_ref, img, floor_coord)
        contours = find_contours(thresh)
        #contours = filter_contours(img_ref, img, contours)

        for contour in contours:
            [x1, y1, x2, y2] = contour
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            """
            cv2.imshow(img_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            """

        dst = save_results(args.repo, img_name, img)

        tp, fp, fn, nb_predict = confusion_matrix(bb=contours,lb=labels)
        accuracy, recall, precision, f1_score = calc_metrics(tp,fp,fn,nb_predict)
        cf_matrix = [[0, fp], [fn, tp]]
        data_results = [accuracy,recall,precision,f1_score]
        # cf_matrix = [[73, 7], [7, 141]]
        # data_results = [92,5,33,12]
        viewer.add_results_image(dst,img_name,cf_matrix,data_results)
    viewer.show_visualization()


if __name__ == "__main__":
    main()

