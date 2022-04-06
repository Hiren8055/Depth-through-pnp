import cv2
import numpy as np
img = cv2.imread('frame0.jpg')
cv2.imshow("before",img)
a = [[[
                      [[
                            399,
                            284,
                            415,
                            347 ] ],
                        [[157,
                            182,
                            167,
                            204]],
                     [[     284,
                            197,
                            298,
                            227]],
                        [[317,
                            218,
                            338,
                            260]],
                    [[
                            138,
                            191,
                            152,
                            220
                        ]],
                        [[
                            150,
                            186,
                            161,
                            211
                        ]
                    ],
                    [
                        [
                            62,
                            223,
                            88,
                            279
                        ]
                    ],
                    [
                        [
                            277,
                            191,
                            288,
                            216
                        ]
                    ],
                    [
                        [
                            343,
                            234,
                            371,
                            292
                        ]
                    ],
                    [
                        [
                            165,
                            179,
                            174,
                            199
                        ]
                    ],
                    [
                        [
                            297,
                            206,
                            314,
                            239
                        ]
                    ],
                    [
                        
                        [
                            122,
                            197,
                            139,
                            231
                        ]
                    ],
                    [[99,
                        207,
                        121,
                        248]], 
                        [[4,
                        252,
                        40,
                        330 ]]
                        
]]]
print(a)
b =len(a[0][0])
top = []
bottom= []
coor=[]
for i in range(b):
    x_bottom = a[0][0][i][0][0]+a[0][0][i][0][2]
    y_botom = a[0][0][i][0][1]+a[0][0][i][0][3]
    top.append([a[0][0][i][0][0],a[0][0][i][0][1]])
    bottom.append([x_bottom,y_botom])
    coor.append([[a[0][0][i][0][0],a[0][0][i][0][1]],[x_bottom,y_botom]])


x = np.zeros(img.shape[:2], dtype="uint8")
for i in range(len(coor)):
    x= cv2.rectangle(x,(a[0][0][i][0][0],a[0][0][i][0][1]),(a[0][0][i][0][2],a[0][0][i][0][3]),(255,255,255),-1)
    # x = cv2.rectangle(img,(a[0][0][i][0][0],a[0][0][i][0][1]),(a[0][0][i][0][2],a[0][0][i][0][3]),(0,0,0),1)
    masked = cv2.bitwise_and(img, img, mask=x)
    

cv2.imshow("window",masked)
# cv2.imwrite("./masked.jpg", masked)

cv2.waitKey(0)
cv2.destroyAllWindows()