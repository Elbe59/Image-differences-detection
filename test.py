from os import listdir
from skimage.metrics import structural_similarity
import numpy as np
import cv2
from _main import *
import pandas as pd
import json
import torch
import torchvision.ops.boxes as bops


a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(a[2:5])

"""
def isOverlapping(bb1, bb2, seuil):
    if bops.box_iou(torch.tensor([bb1]), torch.tensor([bb2])).item() > seuil:
        return True
    else:
        return False

print(isOverlapping([5, 5, 10, 10], [12, 12, 20, 20], 0))


def confusion_matrix(bb, lb):
    seuil = 0.5
    bb_count = [False] * len(bb)
    lb_count = [False] * len(lb)

    for i in range(len(bb)):
        for j in range(len(lb)):
            if isOverlapping(bb[i], lb[j], seuil):
                bb_count[i] = True
                lb_count[i] = True

    tp = sum(bb_count)
    fp = len(bb_count) - sum(bb_count)
    fn = len(lb_count) - sum(lb_count)
    nb_predict = len(bb)

    return tp, fp, fn, nb_predict


def calc_metrics(tp, fp, fn, nb_predict):
    accuracy = tp / nb_predict
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = (2 * precision * recall) / (precision + recall)
    return accuracy, recall, precision, f1_score


bb = [[0, 0, 10, 10], [20, 0, 5, 5]]
lb = [[0, 0, 8, 8], [0, 20, 5, 5]]
tp, fp, fn, nb_predict = confusion_matrix(bb, lb)
accuracy, recall, precision, f1_score = calc_metrics(tp, fp, fn, nb_predict)

print("tp = ", tp)
print("fp = ", fp)
print("fn = ", fn)
print("nb_predict = ", nb_predict)
print("accuracy = ", accuracy)
print("recall = ", recall)
print("precision = ", precision)
print("f1_score = ", f1_score)
"""



