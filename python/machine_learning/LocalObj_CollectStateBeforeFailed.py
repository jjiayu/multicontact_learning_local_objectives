# Input Arguments:
# 1: Working Directory (The root folder of the work space)
# 2: RollOutfolder name
# 3: DataSet Folder name
# 4: Pre Processing Mode: 1) "OriginalForm" 2) "Standarization" 3) "MaxAbs" (lies in [-1, 1])
# 5: Scale Factor
# 6: How many steps before failure (start from 1)

# Import Packages
import numpy as np
import os
import pickle

from numpy.core.shape_base import _stack_dispatcher
from multicontact_learning_local_objectives.python.machine_learning.ml_utils import *
from multicontact_learning_local_objectives.python.terrain_create import *
import matplotlib.pyplot as plt  # Matplotlib
import time
import shutil
import sys
from sklearn import preprocessing

# Get Working Directory and dataset type
workingDirectory = sys.argv[1]
RollOutFoldername = sys.argv[2]
print("Working Folder: \n", workingDirectory)

# ------------------------
# Definr RollOut Path
rolloutPath = workingDirectory + '/' + \
    RollOutFoldername  # "/CleanTrainingSetRollOuts/"

# Create Folder to store datasets
# make the folder if we dont have one
DataSetFolderPath = workingDirectory + "/DataSet"
if not (os.path.isdir(DataSetFolderPath)):
    os.mkdir(DataSetFolderPath)

# DataSetPath
# make the folder if we dont have one
DataSetPath = DataSetFolderPath + "/" + sys.argv[3] + "/"
if not (os.path.isdir(DataSetPath)):
    os.mkdir(DataSetPath)
print("DataSetPath: ", DataSetPath)

# Get Preprocessing Mode
# Can be 1) "OriginalForm" 2) "Standarization" 3) "MaxAbs" (lies in [-1, 1])
PreProcessingMode = sys.argv[4]
print("Pre-Procesing Mode: ", PreProcessingMode)

# -----------------------
# Define Frame Transformation and DataPoint Representation
# Shift world frame type: StanceFoot, (Abandon) InitCoM,  (Abandon) InitSurfBorder, None
Shift_World_Frame_Type = "StanceFoot"
print("Shift World Frame to:\n", Shift_World_Frame_Type)
# Contact Location Representation Type 1) 3DPoints 2) ConvexCombination 3) FollowRectangelBorder
Contact_Representation_Type = "FollowRectangelBorder"
print("Contact Location Representation Type:\n ", Contact_Representation_Type)
# Scaling Vectors? 1 menas no scaling (Now scaling does not make much diff)
if PreProcessingMode == "OriginalForm":
    ScaleFactor = float(sys.argv[5])
    print("Scaling Factor of all quantities (Except Left and Right Swing Flag):\n", ScaleFactor)
else:
    ScaleFactor = 1.0

# Give some time to show the working directory
time.sleep(7)

# ---------------
# Start Collecting Data Points
# get all the file names
filenames = os.listdir(rolloutPath)

# Initialize Dataset
x_all = []
y_all = []

total_file_num = 0
failed_file_num = 0

# loop all the files
for filename in filenames:
    if ".p" in filename:  # a data file
        print("Process: ", filename)

        total_file_num = total_file_num + 1  # get a file

        # Load data
        with open(rolloutPath + '/' + filename, 'rb') as f:
            data = pickle.load(f)

        # print Terrain Settings
        # print(data["TerrainSettings"])

        if not(len(data["SingleOptResultSavings"]) == data["Num_of_Rounds"]):

            print("Failed Tracking RollOuts")
            failed_file_num = failed_file_num + 1

            roundNum = len(data["SingleOptResultSavings"]) - int(sys.argv[6])
            print("Total Length of Single Opt Result: ",
                  len(data["SingleOptResultSavings"]))
            print("Round Number of Interest: ", roundNum)

            if roundNum < 0:
                continue

            singleOptResult = data["SingleOptResultSavings"][roundNum]

            # Get data point
            xtemp, ytemp = getDataPoints(SingleOptRes=singleOptResult, Shift_World_Frame=Shift_World_Frame_Type,
                                         ContactRepresentationType=Contact_Representation_Type, VectorScaleFactor=ScaleFactor)  # InitCoM; InitSurfBorder

            x_all.append(xtemp)
            y_all.append(ytemp)

        else:
            print("Failed Computation")

# make data points to become numpy array
x_all = np.array(x_all)
y_all = np.array(y_all)

# Pre Process data if we want
print("----------Pre-Processing---------------")
#   Compute Scaler
if PreProcessingMode == "OriginalForm":
    print("Keep the original form")
    scaler_X = None
    scaler_Y = None
elif PreProcessingMode == "Standarization":
    print("Standarize the data")
    scaler_X = preprocessing.StandardScaler().fit(x_all)
    scaler_Y = preprocessing.StandardScaler().fit(y_all)
elif PreProcessingMode == "MaxAbs":
    print("Scale the data by MaxAbs into [-1, 1]")
    scaler_X = preprocessing.MaxAbsScaler().fit(x_all)
    scaler_Y = preprocessing.MaxAbsScaler().fit(y_all)
else:
    raise Exception("Unknow Pre-Processing Operation")

#   Then Transform data
if PreProcessingMode == "Standarization" or PreProcessingMode == "MaxAbs":
    print("Transforming Data")
    x_all = scaler_X.transform(x_all)
    y_all = scaler_Y.transform(y_all)

# Get NumberofPreviewSteps
NumPreviewSteps = len(data["SingleOptResultSavings"][0]["ContactSurfs"])

# Save DataPoints and Settings
DatasSet_All = {"input":  x_all,
                "output": y_all,
                "PreProcessMode": PreProcessingMode,
                "Scaler_X": scaler_X,
                "Scaler_Y": scaler_Y,
                "Shift_World_Frame_Type": Shift_World_Frame_Type,
                "Contact_Representation_Type": Contact_Representation_Type,
                "VectorScaleFactor": ScaleFactor,
                "NumPreviewSteps": NumPreviewSteps}

print(" ")
print("DataSet Settings:")
print("- Shift_World_Frame_Type: ", DatasSet_All["Shift_World_Frame_Type"])
print("- Contact Location Representation Type: ",
      DatasSet_All["Contact_Representation_Type"])
print("- Vector Scale Factor: ", DatasSet_All["VectorScaleFactor"])
print("- Number of Preview Steps: ", DatasSet_All["NumPreviewSteps"])
print("- Pre Processing Mode: ", DatasSet_All["PreProcessMode"])

print(" ")
print("Summary: ")
print("- Total Number of Rollouts (files): ", total_file_num)
print("- Total Number of Failed Rollouts (files): ", failed_file_num)

# Save the data
pickle.dump(DatasSet_All, open(DataSetPath + "/data"+'.p', "wb"))
