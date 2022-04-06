import numpy as np
import cv2

img = cv2.imread('Capture.jpg')
# winname = "Capture"
# cv2.namedWindow(winname)        # Create a named window
# cv2.moveWindow(winname, 40,30)
# winname1 = "Capture1"
# cv2.namedWindow(winname1)        # Create a named window
# cv2.moveWindow(winname1, 40,30)

imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thrash = cv2.threshold(imgGrey, img.shape[0], img.shape[1], cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

cv2.imshow("winne1", img)
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    #if len(approx) == 3:
    cv2.putText(img, "Triangle", (x, y+10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    # elif len(approx) == 4:
    #     x1 ,y1, w, h = cv2.boundingRect(approx)
    #     aspectRatio = float(w)/h
    #     print(aspectRatio)
    #     if aspectRatio >= 0.95 and aspectRatio <= 1.05:
    #       cv2.putText(img, "square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    #     else:
    #       cv2.putText(img, "rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    # elif len(approx) == 5:
    #     cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    # elif len(approx) == 10:
    #     cv2.putText(img, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
    # else:
    #     cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))


cv2.imshow("winne", img)
cv2.waitKey(0)
cv2.destroyAllWindows()