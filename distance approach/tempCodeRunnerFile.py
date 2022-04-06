 frame1 = frame.copy()
        cv2.circle(frame1,(detections[j][2][0],detections[j][2][1]), 5, (0,0,255), -1)
        cv2.circle(frame1,(detections[j][2][2],detections[j][2][3]), 5, (0,225,255), -1)
        # cv2.imshow("frame1",frame1)

        # print(detections[j][2][1],detections[j][2][3],detections[j][2][0],detections[j][2][2])
        z = frame[detections[j][2][1]:detections[j][2][3],detections[j][2][0]:detections[j][2][2]]


        


        if z.shape[0] != 0 and z.shape[1]!=0:
            
                
            lab = cv2.cvtColor(z,cv2.COLOR_BGR2LAB)
            L,A,B=cv2.split(lab)
            

            _, thresh1 = cv2.threshold(A, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # cv2.imshow('thresh1',thresh1)      
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
            
            # print(extTop,centerBase,extLeft,extRight)

    # cv2.imshow('thresh2',thresh1)      

    # if z.shape[0] != 0 and z.shape[1]!=0:
        # cv2.imshow("image",z)
    
    y+=1