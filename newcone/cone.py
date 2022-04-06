import cv2
import numpy as np
import json 
import imutils


json_filename = "D:/Solethon/perception/keypoint/test0_cones.json"
with open(json_filename) as f:
    json_data = json.load(f)

path = "D:/Solethon/keypoint/newcone/masked2.mp4"
# output_video_path = "D:/Solethon/keypoint/newcone/masked2.mp4"
cap = cv2.VideoCapture(path)

# Laptop camera 
#pt = [(0,100), (-900,450), (600,100), (1500,450)]

# intel camera 
pt = [(0,225), (-1500,500), (600,225), (2100,500)]

i=0


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video

# video = set_saved_video(cap,output_video_path, (416, 416))

while cap.isOpened():
    
    print("cap is open")
    #############################################################################
    ##########################  cone detection  #################################
    #############################################################################
    _, frame = cap.read()
    ######################### isolation of cones################################################
    
    
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # for i in range(len(json_data[1]['log_data'])):
    detections = json_data[1]['log_data'][i]['detections']
    cv2.imshow("win",frame)
    # c=[]
    x = np.zeros(frame.shape[:2], dtype="uint8")
    for j in range(len(detections)):

        
        x= cv2.rectangle(x,(detections[j][2][0:2]),(detections[j][2][2:4]),(255,255,255),-1)
        
        
        masked = cv2.bitwise_and(frame, frame, mask=x)
        

        masked[detections[j][2][0]:detections[j][2][1],detections[j][2][2]:detections[j][2][3]]
        cv2.imshow("sep",masked)
        # lab = cv2.cvtColor(masked,cv2.COLOR_BGR2LAB)
        # L,A,B=cv2.split(lab)
        # # cv2.imshow('lab',lab) 

            
        # ret, thresh1 = cv2.threshold(A, 120, 255, cv2.THRESH_BINARY + 
        #                                             cv2.THRESH_OTSU)      
        
        # # cv2.imshow('',thresh1)

        # # find contours in thresholded image, then grab the largest
        # # one
        # cnts = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
        #     cv2.CHAIN_APPROX_SIMPLE)
        # cnts = imutils.grab_contours(cnts)
        # c = max(cnts, key=cv2.contourArea)
        # # print(c)
        # # determine the most extreme points along the contour
        # extLeft = tuple(c[c[:, :, 0].argmin()][0])
        # extRight = tuple(c[c[:, :, 0].argmax()][0])
        # extTop = tuple(c[c[:, :, 1].argmin()][0])
        # extBot = tuple(c[c[:, :, 1].argmax()][0])
        # rng = int((extBot[1] - extTop[1]) * 0.2)

        # finalYBot = extBot[1] + rng
        # finalYTop = extBot[1] - rng

        # newC = []
        # for i in c:
        #     if i[0][1] >= finalYTop and i[0][1] <= finalYBot  : 
        #         newC.append((i[0][0],i[0][1]))
        

        # extLeft = min(newC)
        # extRight = max(newC)

        # cv2.drawContours(x, [c], -1, (0, 255, 255), 2)
        # cv2.circle(masked, extLeft, 5, (0, 0, 255), -1)
        # cv2.circle(masked, extRight, 5, (0, 255, 0), -1)
        # cv2.circle(masked, extTop, 5, (0, 0, 255), -1)
        # cv2.circle(masked, extBot, 8, (255, 255, 0), -1)


        # centerBase = ((extLeft[0]+extRight[0])//2 ,(extLeft[1]+extRight[1])//2 )
        # cv2.circle(masked, centerBase, 1, (255, 255, 255), -1)


        # print("top", extTop )
        # print("base center",centerBase)
        # print("left",extLeft)
        # print("right",extRight)
        # show the output image
        

        
        
        
    # cv2.imshow('keypoints',masked)
    i+=1
    # print(len(c))
    # for i in range(len(c)):
    #     cv2.imshow("c",c[i])
    ind_rec = frame[detections[j][2][0]:detections[j][2][1],detections[j][2][2]:detections[j][2][3]]
    if ind_rec.all() != None:
         # print(ind_rec)
        cv2.imshow('rec',ind_rec)
    #video.write(masked)
#####################################################################################################
#############################convex hull#######################################################
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
    blur = cv2.blur(gray, (3, 3)) # blur the image
    ret, thresh = cv2.threshold(blur, 105, 255, cv2.THRESH_BINARY)
    cv2.imshow("Win",thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # create hull array for convex hull points
    hull = []

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))
    # cv2.imshow("Win2",hull)
    # create an empty black image
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)

    # draw contours and hull points
    for i in range(len(contours)):
        color_contours = (0, 255, 0) # green - color for contours
        color = (255, 255, 255) # blue - color for convex hull
        # draw ith contour
        b =cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        a =cv2.drawContours(drawing, hull, i, color, 1, 8)
        cv2.imshow("drawing",b)


    #############################################################################################
    frame = b
    frame = cv2.resize(frame, (600, 450))
    #frame = cv2.imread('coneimg.png')
    img_HSV = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    img_thresh_low = cv2.inRange(img_HSV, np.array([0, 135, 135]),np.array([15, 255, 255]))  # everything that is included in the "left red"
    # cv2.imshow("img_thresh_low",img_thresh_low)
    # img_thresh_high = cv2.inRange(img_HSV, np.array([159, 135, 135]), np.array([179, 255, 255]))  # everything that is included in the "right red"
    # cv2.imshow("img_thresh_high",img_thresh_high)   
    # img_thresh_mid = cv2.inRange(img_HSV, np.array([100, 150, 0]),np.array([140, 255, 255]))  # everything that is included in the "right red"
    # cv2.imshow("img_thresh_mid",img_thresh_mid)  
    # img_thresh = cv2.bitwise_or(img_thresh_low, img_thresh_mid)  # combine the resulting image
    # cv2.imshow("img_thresh",img_thresh)  
    # img_thresh = cv2.bitwise_or(img_thresh, img_thresh_high)
    # cv2.imshow("img_thresh1",img_thresh)
    img_thresh = img_thresh_low

    kernel = np.ones((5, 5))
    # img_thresh_opened = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("img_thresh_opened",img_thresh_opened)
    
    # img_thresh_blurred = cv2.medianBlur(img_thresh_opened, 5)
    # cv2.imshow("img_thresh_blurred",img_thresh_blurred)
    img_thresh_blurred = img_thresh_low

    img_edges = cv2.Canny(img_thresh_blurred, 80, 160)
    cv2.imshow("img_edges",img_edges)
    contours, _ = cv2.findContours(np.array(img_edges), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_contours = np.zeros_like(img_edges)
    

    p =cv2.drawContours(img_contours, contours, -1, (255, 255, 255), 2)
    # cv2.imshow("p",p)
    
    approx_contours = []

    for c in contours:
        approx = cv2.approxPolyDP(c, 10, closed=True)
        approx_contours.append(approx)
    img_approx_contours = np.zeros_like(img_edges)
    q = cv2.drawContours(img_approx_contours, approx_contours, -1, (255, 255, 255), 1)
    # cv2.imshow("q",q)

    all_convex_hulls = []
    for ac in approx_contours:
        all_convex_hulls.append(cv2.convexHull(ac))
    img_all_convex_hulls = np.zeros_like(img_edges)
    r = cv2.drawContours(img_all_convex_hulls, all_convex_hulls, -1, (255, 255, 255), 2)
    # cv2.imshow("r",r)

    convex_hulls_3to10 = []
    for ch in all_convex_hulls:
        if 3 <= len(ch) <= 10:
            convex_hulls_3to10.append(cv2.convexHull(ch))
    img_convex_hulls_3to10 = np.zeros_like(img_edges)
    
    s = cv2.drawContours(img_convex_hulls_3to10, convex_hulls_3to10, -1, (255, 255, 255), 2)  #the result is directly declare in the first parameter img_convex_hulls_3to10
    
    
    
    # cv2.imshow("s",s)



    def convex_hull_pointing_up(ch):
        '''Determines if the path is directed up.
        If so, then this is a cone. '''

        # contour points above center and below

        points_above_center, points_below_center = [], []

        x, y, w, h = cv2.boundingRect(ch)  # coordinates of the upper left corner of the describing rectangle, width and height
        aspect_ratio = w / h  # ratio of rectangle width to height

        # if the rectangle is narrow, continue the definition. If not, the circuit is not suitable
        if aspect_ratio < 0.8:
    # We classify each point of the contour as lying above or below the center	
            vertical_center = y + h / 2

            for point in ch:
                if point[0][
                    1] < vertical_center:  # if the y coordinate of the point is above the center, then add this point to the list of points above the center
                    points_above_center.append(point)
                elif point[0][1] >= vertical_center:
                    points_below_center.append(point)

            # determine the x coordinates of the extreme points below the center
            left_x = points_below_center[0][0][0]
            right_x = points_below_center[0][0][0]
            for point in points_below_center:
                if point[0][0] < left_x:
                    left_x = point[0][0]
                if point[0][0] > right_x:
                    right_x = point[0][0]

            # check if the upper points of the contour lie outside the "base". If yes, then the circuit does not fit
            for point in points_above_center:
                if (point[0][0] < left_x) or (point[0][0] > right_x):
                    return False
        else:
            return False
 
        return True


    cones = []
    bounding_rects = []
    for ch in convex_hulls_3to10:
        if convex_hull_pointing_up(ch):
            cones.append(ch)
            rect = cv2.boundingRect(ch)
            bounding_rects.append(rect)

    img_res = frame.copy()
    # img_res = masked.copy()
    # img_res = b.copy()

    cv2.drawContours(img_res, cones, -1, (255, 255, 255), 2)
    transf = np.zeros([450, 600, 3])

    mybox = []
    pts1 = np.float32([pt[0],pt[1],pt[2],pt[3]])
    pts2 = np.float32([[0,0],[0,450],[600,0],[600,450]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    transf = np.zeros([450, 600, 3])
    # print(bounding_rects)
    for rect in bounding_rects:
        print('previous', rect[0], rect[1], rect[2], rect[3])
        cv2.rectangle(img_res, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (1, 255, 1), 6)
        cv2.circle(img_res,(rect[0], rect[1]), 5, (0,200,255), -1)
        cv2.circle(img_res,(rect[0] + rect[2], rect[1] + rect[3]), 5, (0,200,255), -1)
        cv2.circle(img_res,(rect[0] + rect[2]//2, rect[1] + rect[3]), 5, (255,0,255), -1)
        a = np.array([[(rect[0] + rect[2]), (rect[1] + rect[3])]], dtype='float32')
        a = np.array([a])

        pointsOut = cv2.perspectiveTransform(a, M)
        box = pointsOut[0][0][0], pointsOut[0][0][1]
        mybox.append(box)
        cv2.circle(img_res,((rect[0] + rect[2]//2), (rect[1] + rect[3])), 5, (0,0,255), -1)
        dst2 = cv2.warpPerspective(img_res,M,(600,450), flags=cv2.INTER_LINEAR)
        # print(dst2)
        # cv2.circle(dst2,box, 5, (0,225,255), -1)
        # cv2.circle(transf,box, 5, (0,225,255), -1)


    dst2 = cv2.warpPerspective(img_res,M,(600,450), flags=cv2.INTER_LINEAR)
    #############################################################################



    #############################################################################
    ####################### inverse perspective transform   #####################
    #############################################################################

    img = cv2.resize(img_res, (600, 450))
    rows,cols,channels = img.shape

    #cv2.circle(transf,pt[0], 5, (0,0,255), -1) 	# Filled
    #cv2.circle(transf,pt[1], 5, (0,0,255), -1) 	# Filled
    #cv2.circle(transf,pt[2], 5, (0,0,255), -1) 	# Filled
    #scv2.circle(transf,pt[3], 5, (0,0,255), -1) 	# Filled

    cv2.circle(img,pt[0], 5, (0,0,255), -1) 	# Filled
    cv2.circle(img,pt[1], 5, (0,0,255), -1) 	# Filled
    cv2.circle(img,pt[2], 5, (0,0,255), -1) 	# Filled
    cv2.circle(img,pt[3], 5, (0,0,255), -1) 	# Filled

    #pts1 = np.float32([[30,111],[34,326],[561,53],[554,381]])

    dst = cv2.warpPerspective(img,M,(600,450), flags=cv2.INTER_LINEAR)

    cv2.imshow("MASKED",masked)
    cv2.imshow('image',img_res)
    # cv2.imshow('transform', dst2)
    # cv2.imshow('coordinates', transf)

    #############################################################################

    key = cv2.waitKey(100)
    if key == 27:
        break
    # except:
    #     print("detections not found")
    #     break

## Close and exit
cap.release()
#out.release()
cv2.destroyAllWindows()
# video.release()
print("it's done")
