from matplotlib import pyplot as plt


def connectpoints(l):
    fig = plt.figure(figsize=(20,20), dpi=90)
    # l = [[[1,2,3],[4,5,6],[7,8,9]],[[3,0,5],[6,2,8],[9,4,12]]]

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
        
        ax.plot3D([x1,x2,x3], [y1, y2, y3], [z1, z2, z3], 'red', linewidth=2)
            
        ax.disable_mouse_rotation()
        # ax.view_init(240, -90)    
        
        plt.axis('off')
        # plt.show()
        plt.pause(1)
        plt.clf()

connectpoints()



            





