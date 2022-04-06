import cv2
import numpy as np
img = cv2.imread('frame0.jpg')
cv2.imshow("before",img)
print(np.shape(img))
x = np.zeros(img.shape[:2], dtype="uint8")
x= cv2.rectangle(x,(100,100),(200,200),(255,255,255),-1)
masked = cv2.bitwise_and(img, img, mask=x)
cv2.imshow("window",masked)
cv2.waitKey(0)
cv2.destroyAllWindows()

# mask = np.zeros(image.shape[:2], dtype="uint8")
# cv2.circle(mask, (145, 200), 100, 255, -1)
# masked = cv2.bitwise_and(image, image, mask=mask)
# # show the output images
# cv2.imshow("Circular Mask", mask)
# cv2.imshow("Mask Applied to Image", masked)
# cv2.waitKey(0)