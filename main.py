import os.path
import argparse
import cv2
import numpy as np

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("image1", help="The path of image 1")
parser.add_argument("image2", help="The path of image 2")

args = parser.parse_args()

# Load images
img1 = cv2.imread(os.path.abspath(args.image1))
img2 = cv2.imread(os.path.abspath(args.image2))

# Process
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
diff = cv2.absdiff(gray1, gray2)
blur = cv2.GaussianBlur(diff, (5, 5), 0)
_, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
erode = cv2.erode(thresh, kernel, iterations=5)
dilated = cv2.dilate(erode, kernel, iterations=50)
contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) < 900:
        continue

    (x, y, w, h) = cv2.boundingRect(contour)

    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 10)
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 10)

# Show the spotted differences
height, width, _ = img1.shape
x = np.zeros((height, int(width * 0.01), 3), np.uint8)
result = np.hstack((img1, x, img2))
cv2.namedWindow("Spotted differences", cv2.WINDOW_NORMAL)
cv2.imshow("Spotted differences", result)

# Wait pressed any key for destroy window
cv2.waitKey(0)
cv2.destroyAllWindows()
