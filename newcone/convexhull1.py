import cv2
import numpy as np
#https://learnopencv.com/convex-hull-using-opencv-in-python-and-c/

src = cv2.imread("t2.jpg", 1) # read input image\

# src = src1 [260:330,5:40] 

# cv2.imshow("clipped",src)


gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # convert to grayscale
blur = cv2.blur(gray, (3, 3)) # blur the image
ret, thresh = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY)
#cv2.imshow("Win",thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# create hull array for convex hull points
hull = []

# calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv2.convexHull(contours[i], False))
# cv2.imshow("Win2",hull)
# create an empty black image
drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

# draw contours and hull points
for i in range(len(contours)):
    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 255, 255) # blue - color for convex hull
    # draw ith contour
    b =cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    a =cv2.drawContours(drawing, hull, i, color, 1, 8)
    cv2.imshow("drawing",a)


cv2.waitKey(0) 
cv2.destroyAllWindows()