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
    Description :
    Méthode pour le chargement des images concidérant le répertoire passé en paramètre.
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
    Description :
    Méthode pour le chargement des labels et des coordonnées du masque du sol concidérant le répertoire passé en
    paramètre.
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
    Description :
    Méthode permettant de diviser les valeurs des coordonnées des bounding boxes selon la constantes définie.
    """

    return math.floor(coord / RESIZE_FACTOR)


def resize_up(coord):
    """
    Description :
    Méthode permettant de multiplier les valeurs des coordonnées des bounding boxes par la constantes définie.
    """

    return math.floor(coord * RESIZE_FACTOR)


def save_results(repo, img_name, img):
    """
    Description :
    Méthode pour sauvegarder les images obtenues dans une répertoire ./results
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
    Description :
    Méthode retournant la valeur True si deux bounding boxes bb1 et bb2 se superposent, False sinon. Le seuil permet de
    définir une aire minimum d'overlapping.
    """

    return bops.box_iou(torch.tensor([bb1]), torch.tensor([bb2])).item() > seuil


def sort_overlapping_bb(bb_array):
    """
    Description :
    Méthode permettant de trier les bounding boxes selon leur superposition. Si deux bounding boxes se superposent,
    elles prennent toutes deux pour coordonnées le couple (x1, y1) minimum et (x2, y2) maximum. Les doublons sont
    ensuite supprimés dans une deuxième boucle.
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


def analyze_bb(bb, lb, img):
    """
    Description :
    Cette méthode permet de construire deux liste de booléens bb_count et lb_count. Un True dans bb_count signifie que
    la bounding boxe de même indice dans bb est un TP, un False signifie qu'il s'agit d'un FP. Un False dans lb_count
    signifie que le label de même indice dans lb est un FN.
    """

    bb_count = [False] * len(bb)
    lb_count = [False] * len(lb)

    is_visited = []

    for i in range(len(bb)):
        for j in range(len(lb)):
            if is_overlapping(bb[i], lb[j], SEUIL_IOU):
                bb_count[i] = True
                lb_count[j] = True

    return bb_count, lb_count


def confusion_matrix(bb_count, lb_count):
    """
    Description :
    Cette méthode permet de calculer les nombres de TP, FP et FN à partir des listes bb_count et lb_count retournées
    par la fonction précédente.
    """

    tp = sum(bb_count)
    fp = len(bb_count) - sum(bb_count)
    fn = len(lb_count) - sum(lb_count)
    nb_predict = len(bb_count)

    return tp, fp, fn, nb_predict


def calc_metrics(tp, fp, fn, nb_predict):
    """
    Description :
    Calcule les principales métriques (accuracy, precision, recall, f1-score) à partir des nombres de TP, FP et FN ainsi
    que du nombre de prédictions.
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
    Description :
    Il s'agit ici de la méthode la plus importante de notre programme. Elle permet de réaliser la segmentation
    (ou binarisation) d'une image à l'aide de son image de référence. Les différentes étapes sont décrites en
    commentaires.
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
    Description :
    Cette méthode permet de tracer les contours (ou bounding boxes) à partir d'une image segmentée/binarisée. Les
    bounding boxes de petites tailles sont exclues si elles ne dépassent pas un seuil (une aire) prédéfini.
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
    Description :
    Il s'agit de la routine principale de notre programme regroupant l'ensemble des fonctions précédentes. Les étapes
    sont mentionnées en commentaires.
    """

    # parsing des arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="The path of repository folder. Ex: ressources/Chambre/")
    args = parser.parse_args()

    # loading des images, labels et de l'image de référence
    img_list, img_ref = img_load(os.path.abspath(args.repo))
    labels, floor_coord = labels_load(args.repo)
    viewer.add_original_image(args.repo + "Reference.JPG")

    # boucle principale parcourant les images
    for img_name, img in img_list.items():
        img_ref_resized = cv2.resize(img_ref, DIM_IMG)
        img_resized = cv2.resize(img, DIM_IMG)

        # binarisation et détection des contours
        thresh = process(img_ref_resized, img_resized, floor_coord)
        contours = find_contours(thresh)
        bb_count, lb_count = analyze_bb(bb=contours, lb=labels[img_name], img=img)

        # affichage des FP
        for i in range(len(lb_count)):
            if not lb_count[i]:
                [x1, y1, x2, y2] = labels[img_name][i]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), resize_up(2))

        # affichage des TP
        for j in range(len(bb_count)):
            if bb_count[j]:
                [x1, y1, x2, y2] = contours[j]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), resize_up(2))
        # affichage des FN
            else:
                [x1, y1, x2, y2] = contours[j]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), resize_up(2))

        # calculs de la matrice de confusion et des principales métriques
        tp, fp, fn, nb_predict = confusion_matrix(bb_count, lb_count)
        accuracy, recall, precision, f1_score = calc_metrics(tp, fp, fn, nb_predict)

        # sauvegarde des résultats
        dst = save_results(args.repo, img_name, img)

        # affichage à l'aide de l'interface tkinter
        cf_matrix = [[0, fp], [fn, tp]]
        data_results = [int(accuracy * 100), int(recall * 100), int(precision * 100), int(f1_score * 100)]
        viewer.add_results_image(dst, img_name, cf_matrix, data_results)

    viewer.show_visualization()


if __name__ == "__main__":
    main()
