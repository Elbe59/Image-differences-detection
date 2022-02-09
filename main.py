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
DIM_IMG = (600, 400)
PX = 0

def imgLoad(repo):
    img_list = []

    for im in listdir('./ressources' + repo):
        if im != 'Reference.JPG':
            img_list.append([im, cv2.imread('./ressources' + repo + '/' + im)])
            img_list[-1][1] = cv2.resize(img_list[-1][1], DIM_IMG)

    img_ref = cv2.imread('./ressources' + repo + '/Reference.jpg')
    img_ref = cv2.resize(img_ref, DIM_IMG)

    return img_list, img_ref


def saveResults(repo, img):
    if not os.path.exists('./results' + repo):
        os.makedirs('./results' + repo)
    cv2.imwrite('./results' + repo + '/' + 'RESULT_' + img[0], img[1])


def sortBoundingBoxes(bb_array):
    for cbb in bb_array:
        for bb in bb_array:
            if cbb != bb and isOverlapping(cbb, bb):
                cbb[0] = min(cbb[0], bb[0])
                cbb[1] = min(cbb[1], bb[1])
                cbb[2] = max(cbb[2], bb[2])
                cbb[3] = max(cbb[3], bb[3])


def isOverlapping(cbb, bb):
    if bb[0]-PX <= cbb[0] <= bb[2]+PX and bb[1]-PX <= cbb[1] <= bb[3]+PX:
        return True
    if bb[0]-PX <= cbb[2] <= bb[2]+PX and bb[1]-PX <= cbb[1] <= bb[3]+PX:
        return True
    if bb[0]-PX <= cbb[0] <= bb[2]+PX and bb[1]-PX <= cbb[3] <= bb[3]+PX:
        return True
    if bb[0]-PX <= cbb[2] <= bb[2]+PX and bb[1]-PX <= cbb[3] <= bb[3]+PX:
        return True
    if cbb[0]-PX <= bb[0] <= cbb[2]+PX and cbb[1]-PX <= bb[1] <= cbb[3]+PX:
        return True
    if cbb[0]-PX <= bb[2] <= cbb[2]+PX and cbb[1]-PX <= bb[1] <= cbb[3]+PX:
        return True
    if cbb[0]-PX <= bb[0] <= cbb[2]+PX and cbb[1]-PX <= bb[3] <= cbb[3]+PX:
        return True
    if cbb[0]-PX <= bb[2] <= cbb[2]+PX and cbb[1]-PX <= bb[3] <= cbb[3]+PX:
        return True
    else:
        return False


def threshMask(img, img_ref):
    # RGB -> GREY
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ref_grey = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    # diff
    struct_diff = structural_similarity(img_ref_grey, img_grey, full=True)[1]
    struct_diff = (struct_diff * 255).astype("uint8")

    abs_diff = cv2.absdiff(img_ref_grey, img_grey)

    # Gaussian Blur
    struct_diff = cv2.GaussianBlur(struct_diff, (3, 3), cv2.BORDER_DEFAULT) # à mettre avant diff?
    abs_diff = cv2.GaussianBlur(abs_diff, (3, 3), cv2.BORDER_DEFAULT)

    # Threshold
    struct_thresh = cv2.threshold(struct_diff, 50, 255, cv2.THRESH_BINARY_INV)[1] # adaptiveThreshold
    abs_thresh = cv2.threshold(abs_diff, 50, 255, cv2.THRESH_BINARY)[1]

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
    return final_thresh


def makeBoundingBoxes(thresh):

    edges = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    bb_array = []
    for i in range(len(edges)):
        x, y, w, h = cv2.boundingRect(edges[i])
        bb_array.append([x, y, x+w, y+h])

    sortBoundingBoxes(bb_array)
    #print(bb_array)

    bb_array = cv2.groupRectangles(np.concatenate((bb_array, bb_array)), groupThreshold=1, eps=0.2)[0]
    #print(bb_array)

    return bb_array


def drawBoundingBoxes(img, bb_array):
    for bb in bb_array:
        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 3)

    # affichage img finale
    cv2.imshow("image finale", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img


def main():
    repo = SALON_REPO
    img_list, img_ref = imgLoad(repo)
    for img in img_list:
        thresh = threshMask(img[1], img_ref)
        bb_array = makeBoundingBoxes(thresh)
        final_img = [img[0], drawBoundingBoxes(img[1], bb_array)]
        saveResults(repo, final_img)


if __name__ == "__main__":
    main()
