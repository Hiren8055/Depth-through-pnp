# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
# import matplotlib.animation as animation


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# x =[]
# y =[]
# z =[]
# x_ =[1,2,3,4,5,6,7,8,9,10]
# y_=[5,6,2,3,13,4,1,2,4,8]
# z_=[2,3,3,3,5,7,9,11,9,10]

# data  = [[1,2,3,4,5,6,7,8,9,10],
# [5,6,2,3,13,4,1,2,4,8],
# [2,3,3,3,5,7,9,11,9,10]]
# ani = animation.FuncAnimation(fig, data, len(x_), fargs=(data),
#                                        interval=50, blit=False, repeat=True)

# for i in range(len(x_)):
   
   
#     x.append(x_[i])     
#     y.append(y_[i]) 
#     z.append(z_[i])
#     ax.scatter(x, y, z)



#     plt.show()


#######trial2

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import matplotlib.animation as animation

# fig=plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# xdata, ydata,zdata = [], [],[]
# ln, = plt.plot([], [],[], 'ro')

# def init():
#     ax.set_xlim(0, 2*np.pi)
#     ax.set_ylim(-1, 1)
#     ax.set_zlim(-1, 1)

#     return ln,

# def update(frame):
#     xdata.append(frame)
#     ydata.append(np.sin(frame))
#     zdata.append(frame)
#     ln.set_data(xdata, ydata,zdata)
#     return ln

# ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
#                     init_func=init, blit=True)
                    
# # ani = animation.FuncAnimation(fig, update, len(xdata), fargs=(ln,update),
#                                         # interval=50, blit=False, repeat=True)
# plt.show()

#trial3

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

data  = np.array[[1,2,3,4,5,6,7,8,9,10],[5,6,2,3,13,4,1,2,4,8],[2,3,3,3,5,7,9,11,9,10]]
def animate_scatters(iteration, data, scatters):
    """
    Update the data held by the scatter plot and therefore animates it.
    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)
    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])
    return scatters

    
def main(data, save=True):
    """
    Creates the 3D figure and animates it with the input data.
    Args:
        data (list): List of the data positions at each iteration.
        save (bool): Whether to save the recording of the animation. (Default to False).
    """

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Initialize scatters
    scatters = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:]) for i in range(data[0].shape[0]) ]

    # Number of iterations
    iterations = len(data)

    # Setting the axes properties
    ax.set_xlim3d([-50, 50])
    ax.set_xlabel('X')

    ax.set_ylim3d([-50, 50])
    ax.set_ylabel('Y')

    ax.set_zlim3d([-50, 50])
    ax.set_zlabel('Z')

    ax.set_title('3D Animated Scatter Example')

    # Provide starting angle for the view.
    ax.view_init(25, 10)

    ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                                       interval=50, blit=False, repeat=True)

    if save:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])
        ani.save('3d-scatted-animated.mp4', writer=writer)

    plt.show()


main(data, save=False)