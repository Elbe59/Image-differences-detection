import argparse
import os

import cv2

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("image1", help="The path of image 1")
parser.add_argument("image2", help="The path of image 2")

args = parser.parse_args()

# Load images
img1 = cv2.imread(os.path.abspath(args.image1))
img2 = cv2.imread(os.path.abspath(args.image2))

# Resize images
img1 = cv2.resize(img1, (600, 400))
img2 = cv2.resize(img2, (600, 400))

# Find frequency of pixels in range between 0 and 255
hist1 = cv2.calcHist([img1], [0], None, [256], [0, 255])
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 255])

# Calculate histograms and normalize it
cv2.normalize(hist1, hist1, 0, 255, cv2.NORM_MINMAX)
cv2.normalize(hist2, hist2, 0, 255, cv2.NORM_MINMAX)

# Find metric value
metric = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

print(metric)
