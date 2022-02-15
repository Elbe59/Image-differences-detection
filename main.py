import math
import os
from os import listdir

from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
import numpy as np
import cv2

# --- Constantes ---
CHAMBRE_REPO = '/Chambre'
CUISINE_REPO = '/Cuisine'
SALON_REPO = '/Salon'
DIM_IMG = (600, 400)
PX = 3  # Value to increase the area of the rectangle
fig = plt.figure(figsize=(12,6))
fig.patch.set_facecolor('silver')

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""
    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

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
            if cbb != bb and isOverlappingNew(cbb, bb):
                cbb[0] = min(cbb[0], bb[0])
                cbb[1] = min(cbb[1], bb[1])
                cbb[2] = max(cbb[2], bb[2])
                cbb[3] = max(cbb[3], bb[3])


def isOverlapping(cbb, bb):
    if bb[0] - PX <= cbb[0] <= bb[2] + PX and bb[1] - PX <= cbb[1] <= bb[3] + PX:
        return True
    if bb[0] - PX <= cbb[2] <= bb[2] + PX and bb[1] - PX <= cbb[1] <= bb[3] + PX:
        return True
    if bb[0] - PX <= cbb[0] <= bb[2] + PX and bb[1] - PX <= cbb[3] <= bb[3] + PX:
        return True
    if bb[0] - PX <= cbb[2] <= bb[2] + PX and bb[1] - PX <= cbb[3] <= bb[3] + PX:
        return True
    if cbb[0] - PX <= bb[0] <= cbb[2] + PX and cbb[1] - PX <= bb[1] <= cbb[3] + PX:
        return True
    if cbb[0] - PX <= bb[2] <= cbb[2] + PX and cbb[1] - PX <= bb[1] <= cbb[3] + PX:
        return True
    if cbb[0] - PX <= bb[0] <= cbb[2] + PX and cbb[1] - PX <= bb[3] <= cbb[3] + PX:
        return True
    if cbb[0] - PX <= bb[2] <= cbb[2] + PX and cbb[1] - PX <= bb[3] <= cbb[3] + PX:
        return True
    else:
        return False


def isOverlappingNew(boundingBox1, boundingBox2):
    """
    If one of the extremity of the first bounding box overlap one of the extremity of the second bounding box, the two bounding boxes are overlapping.
    Using the global value of PX which makes the bounding bo larger
    :param boundingBox1: The first bounding box - Rectangle [x1,y1,x2,y2] with (x1,y1) the coordinates of the left-bottom corner and (x2,y2) the coordinated of the right top corner.
    :param boundingBox2: The second bounding box - Rectangle [x1,y1,x2,y2] with (x1,y1) the coordinates of the left-bottom corner and (x2,y2) the coordinated of the right top corner.
    :return: True if the two bounding boxes are overlapping. Else otherwise
    """
    new_boundingBox1 = [boundingBox1[0] - PX, boundingBox1[1] - PX, boundingBox1[2] + PX, boundingBox1[3] + PX]
    new_boundingBox2 = [boundingBox2[0] - PX, boundingBox2[1] - PX, boundingBox2[2] + PX, boundingBox2[3] + PX]
    if (new_boundingBox2[0] >= new_boundingBox1[2]) or (new_boundingBox2[2] <= new_boundingBox1[0]) or (
            new_boundingBox2[3] <= new_boundingBox1[1]) or (new_boundingBox2[1] >= new_boundingBox1[3]):
        return False
    else:
        return True


def threshMask(img, img_ref):
    # RGB -> GREY
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ref_grey = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    # diff
    struct_diff = structural_similarity(img_ref_grey, img_grey, full=True)[1]
    struct_diff = (struct_diff * 255).astype("uint8")

    abs_diff = cv2.absdiff(img_ref_grey, img_grey)

    # Gaussian Blur
    struct_diff = cv2.GaussianBlur(struct_diff, (3, 3), cv2.BORDER_DEFAULT)  # à mettre avant diff?
    abs_diff = cv2.GaussianBlur(abs_diff, (3, 3), cv2.BORDER_DEFAULT)

    # Threshold
    struct_thresh = cv2.threshold(struct_diff, 50, 255, cv2.THRESH_BINARY_INV)[1]  # adaptiveThreshold
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

def adaptative_threshMask(img, img_ref):
    # RGB -> GREY
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_ref_grey = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

    # diff
    struct_diff = structural_similarity(img_ref_grey, img_grey, full=True)[1]
    struct_diff = (struct_diff * 255).astype("uint8")

    abs_diff = cv2.absdiff(img_ref_grey, img_grey)
    # Apply a bilateral filter in order to reduce noise while keeping the edges sharp:
    struct_diff = cv2.bilateralFilter(struct_diff, 15, 25, 25)
    abs_diff = cv2.bilateralFilter(struct_diff, 15, 25, 25)
    """
    # Gaussian Blur
    struct_diff = cv2.GaussianBlur(struct_diff, (3, 3), cv2.BORDER_DEFAULT)  # à mettre avant diff?
    abs_diff = cv2.GaussianBlur(abs_diff, (3, 3), cv2.BORDER_DEFAULT)

    # Threshold
    struct_thresh = cv2.threshold(struct_diff, 50, 255, cv2.THRESH_BINARY_INV)[1]  # adaptiveThreshold
    abs_thresh = cv2.threshold(abs_diff, 50, 255, cv2.THRESH_BINARY)[1]

    # erode/dilate
    struct_thresh = cv2.erode(struct_thresh, np.ones((3, 3), np.uint8))
    struct_thresh = cv2.dilate(struct_thresh, np.ones((9, 9), np.uint8))

    abs_thresh = cv2.erode(abs_thresh, np.ones((3, 3), np.uint8))
    abs_thresh = cv2.dilate(abs_thresh, np.ones((9, 9), np.uint8))

    # thresh comparaison
    final_thresh = np.bitwise_and(struct_thresh, abs_thresh)
    """

    # Threshold
    struct_thresh = cv2.adaptiveThreshold(struct_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    abs_thresh = cv2.adaptiveThreshold(abs_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # erode/dilate
    struct_thresh = cv2.erode(struct_thresh, np.ones((3, 3), np.uint8))
    struct_thresh = cv2.dilate(struct_thresh, np.ones((9, 9), np.uint8))

    abs_thresh = cv2.erode(abs_thresh, np.ones((3, 3), np.uint8))
    abs_thresh = cv2.dilate(abs_thresh, np.ones((9, 9), np.uint8))


    final_thresh = np.bitwise_and(struct_thresh, abs_thresh)

    # Show the Figure:
    plt.show()
    return final_thresh


def makeBoundingBoxes(thresh):
    edges = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

    bb_array = []
    for i in range(len(edges)):
        x, y, w, h = cv2.boundingRect(edges[i])
        bb_array.append([x, y, x + w, y + h])
    print(bb_array)
    sortBoundingBoxes(bb_array)
    print(bb_array)

    bb_array = cv2.groupRectangles(np.concatenate((bb_array, bb_array)), groupThreshold=1, eps=0.2)[0]
    print(bb_array)

    return bb_array


def drawBoundingBoxes(img, bb_array):
    for bb in bb_array:
        cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (255, 0, 0), 3)

    # affichage img finale
    #cv2.imshow("image finale", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return img


def main():
    repo = SALON_REPO
    img_list, img_ref = imgLoad(repo)
    for img in img_list:
        thresh = threshMask(img[1], img_ref)
        bb_array = makeBoundingBoxes(thresh)
        final_img = [img[0], drawBoundingBoxes(img[1], bb_array)]
        saveResults(repo, final_img)
        show_img_with_matplotlib(img_ref, "Original Image", 1)
        show_img_with_matplotlib(final_img[1], 'RESULT_' + final_img[0], 2)
        # Show the Figure:
        plt.show()



if __name__ == "__main__":
    main()

