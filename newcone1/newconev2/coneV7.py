import cv2
import numpy as np
import json 
import imutils
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
json_filename = "D:/Solethon/keypoint/test0_cones.json"
with open(json_filename) as f:
    json_data = json.load(f)


path ="D:/Solethon/keypoint/newcone/test0.mp4"
cap = cv2.VideoCapture(path)
o = []
p = []

figure, ax = plt.subplots(figsize=(8,6))
line1, = ax.plot(o, p)


y=0 #index to json
K=0 #for one frame
while cap.isOpened():
    K+=1 
    print("cap is open")
    _, frame = cap.read()
    detections = json_data[1]['log_data'][y]['detections']
    
    cv2.imshow("win",frame)
    # la = cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)
    # LL,AA,BB=cv2.split(la)
        
    # cv2.imshow('AA',AA)

    # _, thres = cv2.threshold(AA, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('thres',thres)

    x = np.zeros(frame.shape[:2], dtype="uint8")

    # cv2.imshow("x_outer_crop",frame[399:415,284:347])
    translation_vector_arr =[]
    for j in range(len(detections)):
        # cv2.imshow("x_crop",frame[detections[j][2][1]:detections[j][2][3],detections[j][2][0]:detections[j][2][2]])
        
        

        x= cv2.rectangle(x,(detections[j][2][0:2]),(detections[j][2][2:4]),(255,255,255),-1)
        masked = cv2.bitwise_and(frame, frame, mask=x)
        
        frame1 = frame.copy()
        cv2.circle(frame1,(detections[j][2][0],detections[j][2][1]), 5, (0,0,255), -1)
        cv2.circle(frame1,(detections[j][2][2],detections[j][2][3]), 5, (0,225,255), -1)
        # cv2.imshow("frame1",frame1)

        print(detections[j][2][1],detections[j][2][3],detections[j][2][0],detections[j][2][2])
        z = frame[detections[j][2][1]:detections[j][2][3],detections[j][2][0]:detections[j][2][2]]


        


        if z.shape[0] != 0 and z.shape[1]!=0:
            
                
            lab = cv2.cvtColor(z,cv2.COLOR_BGR2LAB)
            L,A,B=cv2.split(lab)
            

            _, thresh1 = cv2.threshold(A, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cv2.imshow('thresh1',thresh1)      
            cnts = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            
            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
            rng = int((extBot[1] - extTop[1]) * 0.2)

            finalYBot = extBot[1] + rng
            finalYTop = extBot[1] - rng

            newC = []
            for i in c:
                if  i[0][1] >= finalYTop and i[0][1] <= finalYBot  : 
                    newC.append((i[0][0],i[0][1]))  
            

            extLeft = min(newC)
            extRight = max(newC)

            cv2.drawContours(z, [c], -1, (0, 255, 255), 2)
            cv2.circle(z, extLeft, 5, (0, 0, 255), -1)
            cv2.circle(z, extRight, 5, (0, 255, 0), -1)
            cv2.circle(z, extTop, 5, (0, 0, 255), -1)
            cv2.circle(z, extBot, 8, (255, 255, 0), -1)


            centerBase = ((extLeft[0]+extRight[0])//2 ,(extLeft[1]+extRight[1])//2 )
            cv2.circle(z, centerBase, 1, (255, 255, 255), -1)
            
            print(extTop,centerBase,extLeft,extRight)
            im = frame[detections[j][2][1]:detections[j][2][3],detections[j][2][0]:detections[j][2][2]]
            size = im.shape

            area =0.5*(extRight[0]-extLeft[0])*(extLeft[1]+extRight[1]/2)
            distance = area/33.6 
            
            line1.set_xdata(x)
            line1.set_ydata(updated_y)
            # print(centerBase,extTop,extLeft,extRight)
            #2D image points. If you change the image, you need to change vector
            # image_points = np.array([
            #                         (9, 56),
            #                         (15, 3),
            #                         (3, 52),
            #                         (15, 61)
            #                         ], dtype="double")

            # # 3D model points.
            # model_points = np.array([
                                    
            #                             (0.0, 0.0, 0.0),        # bottom
            #                             (0.0, 3.0, 0.0),        # top
            #                             (-0.5, 0.0,0.0),        # Left corner
            #                             (0.5, 0.0, 0.0)         # Right corner

            #                         ])

            # # Camera internals

            # focal_length = size[1]
            # center = (size[1]/2, size[0]/2)
            # camera_matrix = np.array(
            #                         [[focal_length, 0, center[0]],
            #                         [0, focal_length, center[1]],
            #                         [0, 0, 1]], dtype = "double"
            #                         )

            # print("Camera Matrix :\n {0}".format(camera_matrix))

            # dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            # (_, translation_vector,rotation_vector,_) = cv2.solvePnPRansac(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

            # print("Rotation Vector:\n {0}".format(rotation_vector))
            # print("Translation Vector:\n {0}".format(translation_vector))

            # # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # # We use this to draw a line sticking out of the nose

            # (nose_end_point2D, jacobian) = cv2.projectPoints(np.float32([(0.0, 0.0, 500.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

            # # for p in image_points:
                
            # #     cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            # print("nose_end_point2D",nose_end_point2D)
            # p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            # p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            # cv2.circle(im,(int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])), 5, (255, 255, 255), -1)
            # # print(nose_end_point2D)
            # cv2.line(im, p1, p2, (255,0,0), 2)
            
            # # Display image
            # cv2.imshow("Output", im)
            # if translation_vector[0][0] != 0 and translation_vector[0][0]
            # fig = plt.figure()
            # ax= plt.axes(projection='3d')
            # print(round(translation_vector[0][0], 2))
            # print(translation_vector[1], 2)
            # print(round(translation_vector[0][2], 2))
            # cv2.waitKey(0)
            # print("trans1",translation_vector[0][0],translation_vector[1][0],translation_vector[2][0])
            # translation_vector_arr.append([translation_vector[0][0],translation_vector[1][0],translation_vector[2][0]])
            # print("trans",translation_vector_arr)
            # print(y)
    cv2.imshow('thresh2',thresh1)      

    if z.shape[0] != 0 and z.shape[1]!=0:
        # print("translation_vector_arr",translation_vector_arr[0],translation_vector_arr[1],translation_vector_arr[2])
        # for i in range(len(translation_vector_arr)):
        #     ax.scatter(translation_vector_arr[i][0],translation_vector_arr[i][1],translation_vector_arr[i][2])    

        # ax.plot3D(translation_vector[0][0],translation_vector[1][0],translation_vector[2][0])
        
        cv2.imshow("image",z)
       
        
        
        

        cv2.imshow("Output_full", im)
    
    y+=1
    cv2.imshow("frame",frame)
    cv2.imshow("MASKED",masked)
    plt.show()
    key = cv2.waitKey(100)
    if key == 27:
        break




# Read Image
