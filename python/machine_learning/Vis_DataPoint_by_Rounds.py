
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import pickle

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1000000

# Load Datta set
# Dataset filename
datasetfile_original = "/home/jiayu/Desktop/MLP_DataSet/Rubbles/DataSet/TrainingSet_OriginalForm/data.p"

dataset_original = pickle.load(open(datasetfile_original, "rb"))

print("World Frame Shift: ", dataset_original["Shift_World_Frame_Type"])
print("Contact Location Representation Type: ",
      dataset_original["Contact_Representation_Type"])
print("Scaling Factor of Variables: ", dataset_original["VectorScaleFactor"])
print("Number of Preview Steps: ", dataset_original["NumPreviewSteps"])

# Test Train Split, for now No Test data
#x_train, x_test, y_train, y_test = train_test_split(dataset["input"], dataset["output"], test_size = 0.01)
X_original = dataset_original["input"]
y_original = dataset_original["output"]
# print(X.max(axis=0))
# print(X.min(axis=0))

X_Round_1 = X_original[0::30, :]
X_Round_2 = X_original[1::30, :]
X_Round_3 = X_original[2::30, :]
X_Round_4 = X_original[3::30, :]
X_Round_5 = X_original[4::30, :]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_Round_1[:, 0], X_Round_1[:, 1], X_Round_1[:, 2],
           c='g', marker='o', linewidth=0.1, alpha=0.025)
ax.scatter(X_Round_2[:, 0], X_Round_2[:, 1], X_Round_2[:, 2],
           c='r', marker='o', linewidth=0.1, alpha=0.025)
ax.scatter(X_Round_3[:, 0], X_Round_3[:, 1], X_Round_3[:, 2],
           c='b', marker='o', linewidth=0.1, alpha=0.025)
ax.scatter(X_Round_4[:, 0], X_Round_4[:, 1], X_Round_3[:, 2],
           c='g', marker='o', linewidth=0.1, alpha=0.025)
ax.scatter(X_Round_5[:, 0], X_Round_5[:, 1], X_Round_5[:, 2],
           c='y', marker='o', linewidth=0.1, alpha=0.025)
ax.set_title("CoM in Local Frame")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.show()
