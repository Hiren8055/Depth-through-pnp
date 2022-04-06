import numpy as np
import cv2 as cv
import glob
import math

# Load previously saved data
# with np.load('CameraParams.npz') as file:
#     , , rvecs, tvecs = [file[i] for i in ('cameraMatrix','dist','rvecs','tvecs')]
mtx = np.array([[1.83008964e+03 ,0.00000000e+00 ,9.46667494e+02],
 [0.00000000e+00 ,1.81224885e+03 ,5.25905682e+02],
 [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]])
dist = np.array([[ 0.1655118 , -1.24365597 , 0.00388576 ,-0.00596596,  0.67328279]])
def draw(img, corners, imgpts):

    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 10)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 10)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 10)

    return img


def drawBoxes(img, corners, imgpts):

    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img



criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
axisBoxes = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

image ="50cm.jpg"
img = cv.imread(image)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, (7,7),None)

# if ret == True:

corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)

# Find the rotation and translation vectors.
ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

# Project 3D points to image plane
imgpts, jac = cv.projectPoints(axisBoxes, rvecs, tvecs, mtx, dist)
print(image,"rotation",rvecs,"\ntranslation",tvecs,"\ndistance :",math.sqrt(tvecs[0]**2+tvecs[1]**2+tvecs[2]**2))

img = drawBoxes(img,corners2,imgpts)
cv.imshow('img',img)

k = cv.waitKey(0) & 0xFF
# if k == ord('q'):
#     break
if k == ord('s'):
    cv.imwrite('pose'+image, img)
# cv.waitKey(0)
# vid.release()
# Destroy all the windows
cv.destroyAllWindows()








############ for video ##########
# vid = cv.VideoCapture("chess1.mp4")
# i = 0
# while(True):
#     i +=1  
#     # Capture the video frame
#     # by frame
#     ret, img = vid.read()

# # for image in glob.glob('undistorted*.png'):

#     # img = cv.imread(image)
#     gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#     ret, corners = cv.findChessboardCorners(gray, (7,7),None)
    
#     # if ret == True:

#     corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)

#     # Find the rotation and translation vectors.
#     ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

#     # Project 3D points to image plane
#     imgpts, jac = cv.projectPoints(axisBoxes, rvecs, tvecs, mtx, dist)
#     print(i,"rotation",rvecs,"\ntranslation",tvecs,"\ndistance :",math.sqrt(tvecs[0]**2+tvecs[1]**2+tvecs[2]**2))

#     img = drawBoxes(img,corners2,imgpts)
#     cv.imshow('img',img)

#     k = cv.waitKey(33) & 0xFF
#     if k == ord('q'):
#         break
#     # if k == ord('s'):
#     #     cv.imwrite('pose'+image, img)

# vid.release()
# # Destroy all the windows
# cv.destroyAllWindows()