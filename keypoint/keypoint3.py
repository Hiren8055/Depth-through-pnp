import numpy as np
import cv2




image = cv2.imread('Capture.jpg')
im_width = image.shape[1]
im_height = image.shape[0]
lower_color_bounds = (240, 110, 82)
upper_color_bounds = (250, 110, 82)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
mask = cv2.inRange(image,lower_color_bounds,upper_color_bounds )
mask_rgb = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
image = image & mask_rgb

cv2.imshow('image1', image) 
   
# Exiting the window if 'q' is pressed on the keyboard.
if cv2.waitKey(0) & 0xFF == ord('q'): 
    cv2.destroyAllWindows()