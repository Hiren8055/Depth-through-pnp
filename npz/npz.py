import numpy as np

# for j in range(9,92):
#     with np.load('D:/Solethon/perception/PnP/calibration/chessCameraParams'+str(j)+'.npz') as file:
#         mtx, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix','dist','rvecs','tvecs')]
#         # print(j,mtx, dist, rvecs, tvecs)
#     print('chessCameraParams'+str(j)+'.npz','\ncamera',mtx,'\ndistortion', dist,'\nrotation', rvecs,'\ntranslation', tvecs)

# j=11
# with np.load('D:/Solethon/perception/PnP/calibration/chessCameraParams'+str(j)+'.npz') as file:
#     mtx, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix','dist','rvecs','tvecs')]
#     # print(j,mtx, dist, rvecs, tvecs)
# print('chessCameraParams'+str(j)+'.npz','\ncamera',mtx,'\ndistortion', dist,'\nrotation', rvecs,'\ntranslation', tvecs)


# for i in range(9,36):
#     b = np.load('D:/Solethon/perception/PnP/calibration/chessCameraParams'+str(i)+'.npz')
#     print(i,b['rvecs'])
i=11 
b = np.load('D:/Solethon/perception/PnP/calibration/chessCameraParams'+str(i)+'.npz')
print(i,b['rvecs'])