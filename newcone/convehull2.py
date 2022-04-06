
#https://theailearner.com/2019/12/05/finding-convex-hull-opencv-python/



import cv2
# Load the image
img1 = cv2.imread('t2.JPG')
# img1 = img1 [260:330,5:40]
img1 = img1 [100:416,5:40]
# Convert it to greyscale
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# Threshold the image
ret, thresh = cv2.threshold(img,110,255,cv2.THRESH_BINARY)
# Find the contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# For each contour, find the convex hull and draw it
# on the original image.
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    cv2.drawContours(img1, [hull], -1, (255, 0, 0), 2)
# Display the final convex hull image
cv2.imshow('ConvexHull', img1)
cv2.waitKey(0)