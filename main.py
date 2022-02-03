import math
import os
from os import listdir
from skimage.metrics import structural_similarity
import numpy as np
import cv2


# --- Constantes ---
CHAMBRE_REPO = '/Chambre'
CUISINE_REPO = '/Cuisine'
SALON_REPO = '/Salon'
EXP_RATIO = 0.2
DIM_MASK = (600, 400)
K = 0.1


def imgLoad(repo):
    img_list = []

    for im in listdir('./ressources' + repo):
        if im != 'Reference.JPG':
            img_list.append([im, cv2.imread('./ressources' + repo + '/' + im)])
            img_list[-1][1] = cv2.resize(img_list[-1][1], DIM_MASK)

    img_ref = cv2.imread('./ressources' + repo + '/Reference.jpg')
    img_ref = cv2.resize(img_ref, DIM_MASK)

    return img_list, img_ref


def saveResults(repo, img):
    if not os.path.exists('./results' + repo):
        os.makedirs('./results' + repo)
    cv2.imwrite('./results' + repo + '/' + 'RESULT_' + img[0], img[1])


def drawBoundingBoxes(img, img_ref):
    # RGB -> GREY
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ref_grey = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    # diff
    struct_diff = structural_similarity(img_ref_grey, img_grey, full=True)[1]
    struct_diff = (struct_diff * 255).astype("uint8")

    abs_diff = cv2.absdiff(img_ref_grey, img_grey)

    # Gaussian Blur
    struct_diff = cv2.GaussianBlur(struct_diff, (7, 7), cv2.BORDER_DEFAULT)  # peut-être (5, 5)
    abs_diff = cv2.GaussianBlur(abs_diff, (3, 3), cv2.BORDER_DEFAULT)

    # Threshold
    struct_thresh = cv2.threshold(struct_diff, 50, 255, cv2.THRESH_BINARY_INV)[1]
    abs_thresh = cv2.threshold(abs_diff, 60, 255, cv2.THRESH_BINARY)[1]

    # erode/dilate
    struct_thresh = cv2.erode(struct_thresh, np.ones((3, 3), np.uint8))
    struct_thresh = cv2.dilate(struct_thresh, np.ones((9, 9), np.uint8))

    abs_thresh = cv2.erode(abs_thresh, np.ones((3, 3), np.uint8))
    abs_thresh = cv2.dilate(abs_thresh, np.ones((9, 9), np.uint8))

    # thresh comparaison
    final_thresh = np.bitwise_and(struct_thresh, abs_thresh)

    # affichage
    """
    cv2.imshow("struct_thresh", struct_thresh)
    cv2.imshow("abs_thresh", abs_thresh)
    cv2.imshow("thresh comparison", final_thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    # bounding boxes
    contours = cv2.findContours(final_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

    # affichage img finale
    cv2.imshow("image finale", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img


def main():
    repo = CHAMBRE_REPO
    img_list, img_ref = imgLoad(repo)
    for img in img_list:
        final_img = [img[0], drawBoundingBoxes(img[1], img_ref)]
        saveResults(repo, final_img)


if __name__ == "__main__":
    main()