import numpy as np
import cv2 as cv

src = cv.imread('src.jpg', -1)
cv.imshow("src",src)
cv.waitKey(0)