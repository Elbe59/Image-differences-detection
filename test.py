from os import listdir
from skimage.metrics import structural_similarity
import numpy as np
import cv2
from _main import *

cbb = [0, 0, 10, 10]
bb = [2, 2, 6, 6]

print(isOverlapping(bb, cbb))

