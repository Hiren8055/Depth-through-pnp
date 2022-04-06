import numpy as np
import os
file_object = open('D:/Solethon/perception/PnP/calibration/npz/sample4.txt', 'a')
j=9
 
for files in os.scandir("D:/Solethon/perception/PnP/calibration"):
    
    
    if files.path.endswith('.npz'):
        
        with np.load('chessCameraParams'+str(j)+'.npz') as file:
            
            mtx, dist, rvecs, tvecs = [file[i] for i in ('cameraMatrix','dist','rvecs','tvecs')]
            print(j,"camera",mtx,"\ndistortion", dist,"\nrotation_vec", rvecs,"\ntranslation_vec", tvecs)
            file_object.write(str([j,'chessCameraParams'+str(j)+'.npz',mtx, dist, rvecs, tvecs]))
            j+=1
file_object.close()
