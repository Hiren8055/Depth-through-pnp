import cv2
import numpy as np
# img = cv2.imread('frame0.jpg')
# cv2.imshow("before",img)
cap = cv2.VideoCapture("D:/Solethon/keypoint/newcone/masked2.mp4")
i=0
while cap.isOpened():
    print("its working")
    ret, frame = cap.read()
    cv2.imshow("win",frame)
    cv2.imwrite("D:/Solethon/keypoint/newcone/t2.jpg",frame)
    i+=1
    if i==2:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

## Close and exit
cap.release()
#out.release()
cv2.destroyAllWindows()