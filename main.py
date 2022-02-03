import cv2
import numpy as np
from skimage.metrics import structural_similarity

# --- lecture images ---
"""
img_ref = cv2.imread('./ressources/Chambre/Reference.jpg')
img_ref = cv2.resize(img_ref, (600, 400))
img = cv2.imread('./ressources/Chambre/IMG_6572.jpg')
img = cv2.resize(img, (600, 400))
"""
"""
img_ref = cv2.imread('./ressources/Cuisine/Reference.jpg')
img_ref = cv2.resize(img_ref, (600, 400))
img = cv2.imread('./ressources/Cuisine/IMG_6565.jpg')
img = cv2.resize(img, (600, 400))
"""

img_ref = cv2.imread('./ressources/Salon/Reference.jpg')
img_ref = cv2.resize(img_ref, (600, 400))
img = cv2.imread('./ressources/Salon/IMG_6557.jpg')
img = cv2.resize(img, (600, 400))


img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_ref_grey = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
"""
cv2.imshow("image de base", img_grey)
cv2.imshow("image ref", img_ref_grey)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


diff = cv2.absdiff(img_ref_grey, img_grey)
"""
diff = cv2.absdiff(img_ref, img)
diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
"""
"""
cv2.imshow("diff", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
diff = cv2.GaussianBlur(diff, (3, 3), cv2.BORDER_DEFAULT)

cv2.imshow("blurred_diff", diff)
cv2.waitKey(0)
cv2.destroyAllWindows()

# remplacer par adaptative treshold ?
ret, thresh = cv2.threshold(diff, 80, 255, cv2.THRESH_BINARY)

cv2.imshow("thresh", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

thresh = cv2.erode(thresh, np.ones((3, 3), np.uint8))
thresh = cv2.dilate(thresh, np.ones((9, 9), np.uint8))

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#print("Number of contours:" + str(len(contours)))

for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

cv2.imshow("image finale", img)
cv2.waitKey(0)
cv2.destroyAllWindows()