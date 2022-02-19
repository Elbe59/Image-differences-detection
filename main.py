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
DIM_IMG = (600, 400)
RESIZE_FACTOR = 10
SEUIL_BB_SIZE = 300
SEUIL_IOU = 0
SEUIL_CROSSCORR = 0.8


def img_load(repo):
    """
    Description
    """

    img_list = {}
    repo = os.path.abspath(repo)
    repo = os.path.basename(repo)  # Get last folder name

    for img in listdir('./ressources/' + repo):
        if img != 'Reference.JPG':
            img_list[img] = cv2.imread('./ressources/' + repo + '/' + img)
            # img_list[img] = cv2.resize(img_list[img], DIM_IMG)

    img_ref = cv2.imread('./ressources/' + repo + '/Reference.jpg')
    # img_ref = cv2.resize(img_ref, DIM_IMG)

    return img_list, img_ref


def labels_load(repo):
    """
    Description
    """

    repo = os.path.abspath(repo)
    repo = os.path.basename(repo)  # Get last folder name
    df = pd.read_csv('./ressources/Labels/' + repo + '_labels.csv')
    labels = {}

    # labels BB
    for i in range(len(df.index) - 1):
        if not df["filename"][i] in labels:
            labels[df["filename"][i]] = []

        bb = json.loads(df["region_shape_attributes"][i].replace('""', '"'))

        x = bb["x"]
        y = bb["y"]
        w = bb["x"] + bb["width"]
        h = bb["y"] + bb["height"]

        labels[df["filename"][i]].append([x, y, w, h])

    # floor coordinates
    coord = json.loads(df["region_shape_attributes"][len(df.index) - 1].replace('""', '"'))
    floor_coord = [[resize(coord["all_points_x"][i]), resize(coord["all_points_y"][i])]
                   for i in range(len(coord["all_points_x"]))]

    return labels, floor_coord


def resize(coord):
    """
    Description
    """

    return math.floor(coord / RESIZE_FACTOR)


def resize_up(coord):
    return math.floor(coord * RESIZE_FACTOR)


def save_results(repo, img_name, img):
    """
    Description
    """

    repo = os.path.abspath(repo)
    repo = os.path.basename(repo)  # Get last folder name

    if not os.path.exists('./results/' + repo):
        os.makedirs('./results/' + repo)

    path = './results/' + repo + '/' + 'RESULT_' + img_name
    cv2.imwrite(path, img)

    return path


def is_overlapping(bb1, bb2, seuil):
    """
    Description
    """

    return bops.box_iou(torch.tensor([bb1]), torch.tensor([bb2])).item() > seuil


def sort_overlapping_bb(bb_array):
    """
    Description
    """

    sorted_bb_array = []

    for cbb in bb_array:
        for bb in bb_array:
            if cbb != bb and is_overlapping(cbb, bb, SEUIL_IOU):
                cbb[0] = min(cbb[0], bb[0])
                cbb[1] = min(cbb[1], bb[1])
                cbb[2] = max(cbb[2], bb[2])
                cbb[3] = max(cbb[3], bb[3])

    for cbb in bb_array:
        if cbb not in sorted_bb_array:
            sorted_bb_array.append(cbb)

    return sorted_bb_array


def confusion_matrix(bb, lb, img):
    """
    Description
    """

    bb_count = [False] * len(bb)
    lb_count = [False] * len(lb)

    is_visited = []

    for i in range(len(bb)):
        for j in range(max(len(bb), len(lb))):
            if is_overlapping(bb[i], lb[j], SEUIL_IOU):
                bb_count[i] = True
                lb_count[j] = True

                [x1, y1, x2, y2] = bb[i]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), resize_up(2))

                is_visited.append(lb[j])

    for label in lb:
        if label not in is_visited:
            [x1, y1, x2, y2] = label
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), resize_up(2))

    tp = sum(bb_count)
    fp = len(bb_count) - sum(bb_count)
    fn = len(lb_count) - sum(lb_count)

    nb_predict = tp + fp + fn

    return tp, fp, fn, nb_predict


def calc_metrics(tp, fp, fn, nb_predict):
    """
    Description
    """

    if nb_predict <= 0:
        return 0, 0, 0, 0

    accuracy = tp / nb_predict

    if tp + fn <= 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if tp + fp <= 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if precision + recall <= 0:
        f1_score = 0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)

    return accuracy, recall, precision, f1_score


def process(img, img_ref, floor_coord):
    """
    Description
    """

    # RGB -> GREY
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ref_grey = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    # diff
    _, struct_diff = structural_similarity(img_ref_grey, img_grey, full=True)
    struct_diff = (struct_diff * 255).astype("uint8")

    # apply a bilateral filter in order to reduce noise while keeping the edges sharp:
    struct_diff = cv2.bilateralFilter(struct_diff, 10, 50, 50)

    # threshold
    struct_thresh = cv2.threshold(struct_diff, 80, 255, cv2.THRESH_BINARY_INV)[1]

    # dilate
    final_thresh = cv2.dilate(struct_thresh, np.ones((2, 2), np.uint8))

    # floor mask
    contours = np.array(floor_coord)
    floor_mask = np.zeros((DIM_IMG[1], DIM_IMG[0]), dtype=np.uint8)
    cv2.fillPoly(floor_mask, pts=[contours], color=(255, 255, 255))
    final_thresh = cv2.bitwise_and(final_thresh, final_thresh, mask=floor_mask)

    return final_thresh


def find_contours(thresh):
    """
    Description
    """

    contours = []
    edges, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in edges:
        if cv2.contourArea(contour) < SEUIL_BB_SIZE:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)

        contours.append([resize_up(x), resize_up(y), resize_up(x + w), resize_up(y + h)])

    for i in range(2):
        contours = sort_overlapping_bb(contours)

    return contours


def main():
    """
    Description
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="The path of repository folder. Ex: ressources/Chambre/")
    args = parser.parse_args()

    img_list, img_ref = img_load(os.path.abspath(args.repo))

    labels, floor_coord = labels_load(args.repo)
    viewer.add_original_image(args.repo + "Reference.JPG")

    for img_name, img in img_list.items():
        img_ref_resized = cv2.resize(img_ref, DIM_IMG)
        img_resized = cv2.resize(img, DIM_IMG)

        thresh = process(img_ref_resized, img_resized, floor_coord)
        contours = find_contours(thresh)

        for label in labels[img_name]:
            [x1, y1, x2, y2] = label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), resize_up(1))

        for contour in contours:
            [x1, y1, x2, y2] = contour
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), resize_up(2))

        tp, fp, fn, nb_predict = confusion_matrix(bb=contours, lb=labels[img_name], img=img)
        accuracy, recall, precision, f1_score = calc_metrics(tp, fp, fn, nb_predict)

        dst = save_results(args.repo, img_name, img)

        cf_matrix = [[0, fp], [fn, tp]]
        data_results = [int(accuracy * 100), int(recall * 100), int(precision * 100), int(f1_score * 100)]
        viewer.add_results_image(dst, img_name, cf_matrix, data_results)

    viewer.show_visualization()


if __name__ == "__main__":
    main()
