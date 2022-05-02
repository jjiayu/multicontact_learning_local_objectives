
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pickle

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1000000

# Load Datta set
# Dataset filename
# datasetfile_original = "/home/jiayu/Desktop/MLP_DataSet/Rubbles_TimeTrack/DataSet/TrainingInit/data.p"
# datasetfile_aug = "/home/jiayu/Desktop/MLP_DataSet/Rubbles_TimeTrack/DataSet/TrainingInit/data.p"
# datasetfile_failed_state = "/home/jiayu/Desktop/MLP_DataSet/Rubbles_TimeTrack/DataSet/TrainingInit/data.p"
# datasetfile_new = "/home/jiayu/Desktop/MLP_DataSet/Rubbles_TimeTrack/DataSet/TrainingInit/data.p"


# Dataset filename
datasetfile_original = "/home/jiayu/Desktop/MLP_DataSet/LargeSlope_TimeTrack_Angle_17_26/DataSet/TrainingInit/data.p"
datasetfile_aug = "/home/jiayu/Desktop/MLP_DataSet/LargeSlope_TimeTrack_Angle_17_26/DataSet/TrainingInit/data.p"
datasetfile_failed_state = "/home/jiayu/Desktop/MLP_DataSet/LargeSlope_TimeTrack_Angle_17_26/DataSet/TrainingInit/data.p"
datasetfile_new = "/home/jiayu/Desktop/MLP_DataSet/LargeSlope_TimeTrack_Angle_17_26/DataSet/TrainingInit/data.p"


dataset_original = pickle.load(open(datasetfile_original, "rb"))
dataset_aug = pickle.load(open(datasetfile_aug, "rb"))
dataset_failed_state = pickle.load(open(datasetfile_failed_state, "rb"))
dataset_new = pickle.load(open(datasetfile_new, "rb"))

print("World Frame Shift: ", dataset_original["Shift_World_Frame_Type"])
print("Contact Location Representation Type: ",
      dataset_original["Contact_Representation_Type"])
print("Scaling Factor of Variables: ", dataset_original["VectorScaleFactor"])
print("Number of Preview Steps: ", dataset_original["NumPreviewSteps"])

# Test Train Split, for now No Test data
#x_train, x_test, y_train, y_test = train_test_split(dataset["input"], dataset["output"], test_size = 0.01)
X_original = dataset_original["input"]
y_original = dataset_original["output"]
X_aug = dataset_aug["input"]
y_aug = dataset_aug["output"]
X_failed = dataset_failed_state["input"]
y_failed = dataset_failed_state["output"]
X_new = dataset_new["input"]
y_new = dataset_new["output"]
# print(X.max(axis=0))
# print(X.min(axis=0))
X_original = X_original


# #Clean DataPoints
# OutlierCnt = 0
# Velo_x_lim = 0.5
# Velo_y_lim = 0.5
# Velo_z_lim = 0.225
# Lx_lim = 0.75
# Ly_lim = 0.75
# Lz_lim = 0.35
# for i in range(X_original.shape[0]):
#     X_temp = X_original[i,:]
#     if np.absolute(X_temp[6])>Lx_lim or np.absolute(X_temp[7])>Ly_lim or np.absolute(X_temp[8])>Lz_lim or np.absolute(X_temp[4]) > Velo_y_lim:
#         OutlierCnt = OutlierCnt + 1
#         X_original[i,:]=np.zeros((1,85))
# print(OutlierCnt)

# OutlierCnt = 0
# for i in range(X_aug.shape[0]):
#     X_temp = X_aug[i,:]
#     if np.absolute(X_temp[6])>Lx_lim or np.absolute(X_temp[7])>Ly_lim or np.absolute(X_temp[8])>Lz_lim or np.absolute(X_temp[4]) > Velo_y_lim:
#         X_aug[i,:]=np.zeros((1,85))
#         OutlierCnt = OutlierCnt + 1
# print(OutlierCnt)

# OutlierCnt = 0
# for i in range(X_failed.shape[0]):
#     X_temp = X_failed[i,:]
#     if np.absolute(X_temp[6])>Lx_lim or np.absolute(X_temp[7])>Ly_lim or np.absolute(X_temp[8])>Lz_lim or np.absolute(X_temp[4]) > Velo_y_lim:
#         X_failed[i,:]=np.zeros((1,85))
#         OutlierCnt = OutlierCnt + 1
# print(OutlierCnt)

# OutlierCnt = 0
# for i in range(X_new.shape[0]):
#     X_temp = X_new[i,:]
#     if np.absolute(X_temp[6])>Lx_lim or np.absolute(X_temp[7])>Ly_lim or np.absolute(X_temp[8])>Lz_lim or np.absolute(X_temp[4]) > Velo_y_lim:
#         X_new[i,:]=np.zeros((1,85))
#         OutlierCnt = OutlierCnt + 1
# print(OutlierCnt)


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_original[:, 0], X_original[:, 1], X_original[:,
           2], c='g', marker='o', linewidth=0.1, alpha=0.05)
ax.scatter(X_aug[:, 0], X_aug[:, 1], X_aug[:, 2],
           c='b', marker='o', linewidth=0.1, alpha=0.05)
ax.scatter(X_failed[:, 0], X_failed[:, 1], X_failed[:, 2],
           c='r', marker='o', linewidth=0.1, alpha=0.)
ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2],
           c='k', marker='o', linewidth=0.1, alpha=0.)
ax.set_title("CoM in Local Frame")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_zlim([0.59, 0.81])

plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_original[:, 3], X_original[:, 4], X_original[:,
           5], c='g', marker='o', linewidth=0.1, alpha=0.05)
ax.scatter(X_aug[:, 3], X_aug[:, 4], X_aug[:, 5],
           c='b', marker='o', linewidth=0.1, alpha=0.05)
ax.scatter(X_failed[:, 3], X_failed[:, 4], X_failed[:, 5],
           c='r', marker='o', linewidth=0.1, alpha=0.)
ax.scatter(X_new[:, 3], X_new[:, 4], X_new[:, 5],
           c='k', marker='o', linewidth=0.1, alpha=0.)
ax.set_title("CoM dot in Local Frame")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_original[:, 6], X_original[:, 7], X_original[:,
           8], c='g', marker='o', linewidth=0.1, alpha=0.05)
ax.scatter(X_aug[:, 6], X_aug[:, 7], X_aug[:, 8],
           c='b', marker='o', linewidth=0.1, alpha=0.05)
ax.scatter(X_failed[:, 6], X_failed[:, 7], X_failed[:, 8],
           c='r', marker='o', linewidth=0.1, alpha=0.)
ax.scatter(X_new[:, 6], X_new[:, 7], X_new[:, 8],
           c='k', marker='o', linewidth=0.1, alpha=0.)
ax.set_title("Am in Local Frame")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([-4, 4])
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_original[:, 9], X_original[:, 10], X_original[:,
           11], c='g', marker='o', linewidth=0.1, alpha=0.05)
ax.scatter(X_aug[:, 9], X_aug[:, 10], X_aug[:, 11],
           c='b', marker='o', linewidth=0.1, alpha=0.05)
ax.scatter(X_failed[:, 9], X_failed[:, 10], X_failed[:, 11],
           c='r', marker='o', linewidth=0.1, alpha=0.)
ax.scatter(X_new[:, 9], X_new[:, 10], X_new[:, 11],
           c='k', marker='o', linewidth=0.1, alpha=0.)
ax.set_title("Contact Location in Local Frame")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(y_original[:, 11], y_original[:, 12], y_original[:,
           13], c='g', marker='o', linewidth=0.1, alpha=0.05)
ax.scatter(y_aug[:, 11], y_aug[:, 12], y_aug[:, 13],
           c='b', marker='o', linewidth=0.1, alpha=0.05)
ax.scatter(y_failed[:, 11], y_failed[:, 12], y_failed[:, 13],
           c='r', marker='o', linewidth=0.1, alpha=0.)
ax.scatter(y_new[:, 11], y_new[:, 12], y_new[:, 13],
           c='k', marker='o', linewidth=0.1, alpha=0.)
ax.set_title("Contact Timing Vector")
ax.set_xlabel("1")
ax.set_ylabel("2")
ax.set_zlabel("3")
plt.show()
