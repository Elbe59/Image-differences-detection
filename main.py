import numpy as np
import cv2

# --- lecture images ---
img_ref = cv2.imread('./ressources/Chambre/Reference.jpg', 0)
#img_ref = cv2.imread('./ressources/Cuisine/Reference.jpg', 0)
img_ref = cv2.resize(img_ref, (600, 400))
img = cv2.imread('./ressources/Chambre/IMG_6570.jpg', 0)
#img = cv2.imread('./ressources/Cuisine/IMG_6563.jpg', 0)
img = cv2.resize(img, (600, 500))

# --- affichage images ---
"""
cv2.imshow('Image de ref', img_ref)
cv2.imshow('Image', img)
cv2.waitKey(0)
"""

# --- threshold ---
"""
img_ref = cv2.equalizeHist(img_ref)
img = cv2.equalizeHist(img)
"""

diff = (img - img_ref)**2
diff_blurred = cv2.GaussianBlur(diff, (21, 21), 3)
ret, thresh = cv2.threshold(diff_blurred, 80, 255, cv2.THRESH_BINARY)

cv2.imshow('treshold', thresh)
cv2.waitKey(0)

