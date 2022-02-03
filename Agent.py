from skimage.metrics import structural_similarity
import cv2
import numpy as np


# --- lecture images ---
"""
img_ref = cv2.imread('./ressources/Chambre/Reference.jpg')
img_ref = cv2.resize(img_ref, (600, 400))
img = cv2.imread('./ressources/Chambre/IMG_6571.jpg')
img = cv2.resize(img, (600, 400))
"""

img_ref = cv2.imread('./ressources/Cuisine/Reference.jpg')
img_ref = cv2.resize(img_ref, (600, 400))
img = cv2.imread('./ressources/Cuisine/IMG_6563.jpg')
img = cv2.resize(img, (600, 400))

"""
img_ref = cv2.imread('./ressources/Salon/Reference.jpg')
img_ref = cv2.resize(img_ref, (600, 400))
img = cv2.imread('./ressources/Salon/IMG_6560.jpg')
img = cv2.resize(img, (600, 400))
"""


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


# thresh comparison
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