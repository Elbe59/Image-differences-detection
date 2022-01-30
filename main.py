import os.path

import cv2
import argparse

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("image1", help="The path of image 1")
parser.add_argument("image2", help="The path of image 2")

args = parser.parse_args()

# Load images
img1 = cv2.imread(os.path.abspath(args.image1))
img2 = cv2.imread(os.path.abspath(args.image2))

# Process
diff = cv2.absdiff(img1, img2)
gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 75, 255, cv2.THRESH_BINARY)
dilated = cv2.dilate(thresh, None, iterations=50)
cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
cv2.imshow("Test", thresh)
contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# for contour in contours:
#     (x, y, w, h) = cv2.boundingRect(contour)
#
#     if cv2.contourArea(contour) < 900:
#         continue
#
#     cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 10)
#
# # Create window
# cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
#
# # Show contours on window
# cv2.imshow("Detection", img2)

# Wait pressed any key for destroy window
cv2.waitKey(0)
cv2.destroyAllWindows()
