#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Load Data

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 1000000

# Load Datta set
# Dataset filename
dataset_file = "/home/jiayu/Desktop/MLP_DataSet/Rubbles/DataSet/TrainingSet_OriginalForm_Initial_Full/data.p"

dataset = pickle.load(open(dataset_file, "rb"))

print("World Frame Shift: ", dataset["Shift_World_Frame_Type"])
print("Contact Location Representation Type: ",
      dataset["Contact_Representation_Type"])
print("Scaling Factor of Variables: ", dataset["VectorScaleFactor"])
print("Number of Preview Steps: ", dataset["NumPreviewSteps"])

# Test Train Split, for now No Test data
#x_train, x_test, y_train, y_test = train_test_split(dataset["input"], dataset["output"], test_size = 0.01)
X = dataset["input"]
y = dataset["output"]

# print(X.max(axis=0))
# print(X.min(axis=0))


# In[13]:


pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(principalComponents[:, 0], principalComponents[:, 1],
           principalComponents[:, 2], marker='o', linewidth=0.001, color='b', alpha=0.0025)
plt.show()

# Load Aug dataset
dataset_file = "/home/jiayu/Desktop/MLP_DataSet/Rubbles_Incremental/DataSet/TrainingSet_OrignalForm_Aug_Round1/data.p"
dataset = pickle.load(open(dataset_file, "rb"))
X_aug = dataset["input"]
y_aug = dataset["output"]

principalComponents_aug = pca.fit_transform(X_aug)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(principalComponents[:, 0], principalComponents[:, 1],
           principalComponents[:, 2], marker='o', linewidth=0.001, color='b', alpha=0.0025)
ax.scatter(principalComponents_aug[:, 0], principalComponents_aug[:, 1],
           principalComponents_aug[:, 2], marker='o', linewidth=0.001, color='g', alpha=0.0025)
plt.show()
