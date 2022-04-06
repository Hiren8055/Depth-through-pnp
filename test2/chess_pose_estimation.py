import numpy as np
import cv2 as cv
import glob
import pandas as pd
import os
j=9
image1 = 'chess_'+str(j)+'.jpg'
savename = "chessaxisbox"
npsave = 'chessCameraParams'+str(j)+'.npz'
mtx_array = []
dist_array = []
rvecs_array = []
pnp_rvecs_array = []
tvecs_array = []
pnp_tvecs_array = []

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

# Load previously saved data
for files in os.scandir("D:/Solethon/perception/PnP/calibration"):
    if files.path.endswith('.npz'):
        j+=1
        with np.load(npsave) as file:
            mtx, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix','dist','rvecs','tvecs')]

        
        # print(str(mtx))

            mtx_array.append(mtx.tolist())
            dist_array.append([dist])
            rvecs_array.append([list(rvecs)])
            tvecs_array.append([list(tvecs)])
        

        





# k = 8
# # for image in :
#     k+=1
        img = cv.imread(glob.glob(image1))
        # cv.imshow("img",img)
        
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (7,7),None)
        
        if ret == True:

            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1), criteria)

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

            pnp_rvecs_array.append([rvecs])
            pnp_tvecs_array.append([tvecs])
            

            # print("pnp_rvecs",rvecs)
            # print("pnp_tvecs",tvecs)

            # Project 3D points to image plane
            imgpts, jac = cv.projectPoints(axisBoxes, rvecs, tvecs, mtx, dist)

            img = drawBoxes(img,corners2,imgpts)
            # cv.imshow('img',img)

            # k = cv.waitKey(0) & 0xFF
            #if k == ord('s'):
            cv.imwrite(savename+str(j)+".jpg", img)



        cv.destroyAllWindows()
# print(mtx_array)
# df= pd.DataFrame()
# df["mtx"] = mtx_array
# df["dist"] = dist_array
# df["rvecs"] = rvecs_array
# df["tvecs"] = tvecs_array
# df["pnp_tvecs"] = pnp_tvecs_array
# df["pnp_rvecs"] = pnp_rvecs_array
df.to_csv('df.csv')