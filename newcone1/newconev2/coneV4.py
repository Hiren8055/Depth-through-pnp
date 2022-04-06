import cv2

img_grayscale = cv2.imread('D:\\Solethon\\keypoint\\newcone\\t.jpg')
cv2.imshow("x_crop",img_grayscale[284:347,399:415])


cv2.waitKey(0)

cv2.destroyAllWindows()
