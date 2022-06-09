import numpy as np

#Vicon Marker Coordinates

marker_1 = np.array([[116.66999816894531],[80], [-20.572000503540039]])

marker_2 = np.array([[116.66999816894531],[-80],[-20.572000503540039]])

marker_3 = np.array([[-203.39999389648438],[124.28199768066406],[-113.13999938964844]])

marker_4 = np.array([[-203.39999389648438],[-35.717998504638672],[-113.13999938964844]])

marker_5 = np.array([[-203.39999389648438],[44.282001495361328],[-133.13999938964844]])

#Plotting

import matplotlib.pyplot as plt  # Matplotlib
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
#ax.set_ylim3d(-1, 1)

ax.plot([0.0, 50.0], [0.0,0.0],zs=[0.0,0.0], color = 'r') #x-axis
ax.plot([0.0, 0.0], [0.0,50.0],zs=[0.0,0.0], color = 'g') #y-axis
ax.plot([0.0, 0.0], [0.0,0.0],zs=[0.0,50.0], color = 'b') #z-axis

#Plot markers
ax.scatter(marker_1[0], marker_1[1],
           marker_1[2], c='k', marker='o', linewidth=10)

ax.scatter(marker_2[0], marker_2[1],
           marker_2[2], c='k', marker='o', linewidth=10)

ax.scatter(marker_3[0], marker_3[1],
           marker_3[2], c='k', marker='o', linewidth=10)

ax.scatter(marker_4[0], marker_4[1],
           marker_4[2], c='k', marker='o', linewidth=10)

ax.scatter(marker_5[0], marker_5[1],
           marker_5[2], c='k', marker='o', linewidth=10)

plt.show()