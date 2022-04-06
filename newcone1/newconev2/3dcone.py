from matplotlib import pyplot as plt


def connectpoints():
    fig = plt.figure(figsize=(30,30), dpi=90)
    l = [[[1,2,3,4,5,6,7,8,9,10],[5,6,2,3,13,4,1,2,4,8],[2,3,3,3,5,7,9,11,9,10]]]
    # print(l[1][0][1])
    for arr in l:
        x1 = arr[0][0]
        x2 = arr[1][0]
        x3 = arr[2][0]

        y1 = arr[0][1]
        y2 = arr[1][1]
        y3 = arr[2][1]

        z1 = arr[0][2]
        z2 = arr[1][2]
        z3 = arr[2][2]

       
       
        ax = fig.add_subplot(111, projection="3d")
        for i in range(10):
            ax.scatter3D([l[0][0][i],l[0][1][i],l[0][2][i]])
            # ax.plot3D([1,2,3])    
            ax.disable_mouse_rotation()
            # ax.view_init(240, -90)    
            
            # plt.axis('off')
            # plt.show()
            plt.pause(100)
            plt.clf()

connectpoints()



            





