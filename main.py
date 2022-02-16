import argparse
import math
import os
from os import listdir

from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
import numpy as np
import cv2


def img_load(repo):
    img_list = {}

    for im in listdir(repo):
        if im != 'Reference.JPG':
            img_list[im] = cv2.imread(repo + '/' + im)
            # img_list[im] = cv2.resize(img_list[im], DIM_IMG)

    img_ref = cv2.imread(repo + '/Reference.jpg')
    # img_ref = cv2.resize(img_ref, DIM_IMG)

    return img_list, img_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", help="The path of repository folder")

    args = parser.parse_args()

    img_list, img_ref = img_load(os.path.abspath(args.repo))

    for img in img_list:
        print(img)


if __name__ == "__main__":
    main()
