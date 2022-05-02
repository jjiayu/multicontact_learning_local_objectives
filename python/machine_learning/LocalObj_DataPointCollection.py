# Input Arguments:
# 1: Working Directory (The root folder of the work space)
# 2: RollOutfolder name
# 3: DataSet Folder name
# 4: Pre Processing Mode: 1) "OriginalForm" 2) "Standarization" 3) "MaxAbs" (lies in [-1, 1])
# 5: Scale Factor
# 6: Start Round of for Collecting the Data Points (Counter starts from 0)
# 7: Terminal Round for Collecting the DataPoints (Counter starts from 0)
# 8: Remove Outlier or not 1) Remove3SigmaOutlier 2)KeepOutlier 3) RemovebyClip 4)RemoveMultiGaussainOutlier

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

# Get start and terminal round number
startroundNum = int(sys.argv[6])
terminalroundNum = int(sys.argv[7])
print("Start Collection from Round Number: ", startroundNum)
print("Stop Collection at Round Number:", terminalroundNum)

# Dealing with Outlier
OutlierFlag = sys.argv[8]

# Print Outlier cleaning information
if OutlierFlag == "Remove3SigmaOutlier":
    print("Remove Outlier with 3 Sigma on robot state axis, for both input and output")
    Outlier_Cnt = 0
elif OutlierFlag == "KeepOutlier":
    print("We will keep Outliers")
elif OutlierFlag == "RemovebyClip":
    print("We will remove Outliers by Clipping on the variable boundary")
else:
    raise Exception("Unknown Outlier Flag")

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
success_file_num = 0

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

        if len(data["SingleOptResultSavings"]) == data["Num_of_Rounds"]:

            print("Success Computation")
            success_file_num = success_file_num + 1

            # range(data["Num_of_Rounds"]):
            for roundNum in range(startroundNum, terminalroundNum+1):
                #print("   Process Round: ", roundNum)
                # Get Single Optimization Result of current result
                singleOptResult = data["SingleOptResultSavings"][roundNum]

                # Get data point
                xtemp, ytemp = getDataPoints(SingleOptRes=singleOptResult, Shift_World_Frame=Shift_World_Frame_Type,
                                             ContactRepresentationType=Contact_Representation_Type, VectorScaleFactor=ScaleFactor)  # InitCoM; InitSurfBorder

                # Save datapoints
                x_all.append(xtemp)
                y_all.append(ytemp)

        else:
            print("Failed Computation")

# make data points to become numpy array
x_all = np.array(x_all)
y_all = np.array(y_all)


# -------------------
# Filter out outliers
#   make outlier_idx container
outlier_Idx = []

#   Get Outlier Index
if OutlierFlag == "KeepOutlier":
    print("We keep the outliers")
elif OutlierFlag == "Remove3SigmaOutlier":
    print("Removing Outliers using 3 Sigma on selected axis")
    print("Number of data points before cleaning: ", x_all.shape[0])
    # Compute Mean
    x_mean = np.mean(x_all, axis=0)
    x_std = np.std(x_all, axis=0)

    y_mean = np.mean(y_all, axis=0)
    y_std = np.std(y_all, axis=0)

    for data_idx in range(x_all.shape[0]):
        xdata_Temp = x_all[data_idx]
        ydata_Temp = y_all[data_idx]
        outlierflag = False
        # Scan input
        for axis_num in range(12):
            if xdata_Temp[axis_num] > x_mean[axis_num] + 3*x_std[axis_num] or xdata_Temp[axis_num] < x_mean[axis_num] - 3*x_std[axis_num]:
                #print("Input axis ", axis_num, "is an outlier")
                outlierflag = True
        # Scan output
        for axis_num in range(y_all.shape[1]):
            if ydata_Temp[axis_num] > y_mean[axis_num] + 3*y_std[axis_num] or ydata_Temp[axis_num] < y_mean[axis_num] - 3*y_std[axis_num]:
                #print("Output axis ", axis_num, " is an outlier")
                outlierflag = True
        # Add outlier flag
        if outlierflag == True:
            print("Data Point ", data_idx, "is an outlier")
            outlier_Idx.append(data_idx)

elif OutlierFlag == "RemovebyClip":
    # Loop over all the datapoints
    for data_idx in range(x_all.shape[0]):
        xdata_Temp = x_all[data_idx]
        ydata_Temp = y_all[data_idx]
        Velo_x_lim = 0.5
        Velo_y_lim = 0.5
        Velo_z_lim = 0.225
        Lx_lim = 0.75
        Ly_lim = 0.75
        Lz_lim = 0.35
        # Check if it is an outlier
        if np.absolute(xdata_Temp[3]) > Velo_x_lim or np.absolute(xdata_Temp[4]) > Velo_y_lim or np.absolute(xdata_Temp[5]) > Velo_z_lim or \
           np.absolute(xdata_Temp[6]) > Lx_lim or np.absolute(xdata_Temp[7]) > Ly_lim or np.absolute(xdata_Temp[8]) > Lz_lim or \
           np.absolute(ydata_Temp[3]) > Velo_x_lim or np.absolute(ydata_Temp[4]) > Velo_y_lim or np.absolute(ydata_Temp[5]) > Velo_z_lim or \
           np.absolute(ydata_Temp[6]) > Lx_lim or np.absolute(ydata_Temp[7]) > Ly_lim or np.absolute(ydata_Temp[8]) > Lz_lim:
            print("Data Point ", data_idx, "is an outlier")
            outlier_Idx.append(data_idx)
else:
    raise Exception("Unknown outlier handling method")

#   Delete outlier rows
x_all_clean = np.delete(x_all, outlier_Idx, 0)
y_all_clean = np.delete(y_all, outlier_Idx, 0)
# re point the vectors
x_all = x_all_clean
y_all = y_all_clean

# ---------------
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
print("- Total Number of Success Rollouts (files): ", success_file_num)
print("- Total Number of DataPoint we should collect: ",
      success_file_num*(terminalroundNum-startroundNum+1))
if OutlierFlag == "Remove3SigmaOutlier" or OutlierFlag == "RemovebyClip":
    print("- Outliersthat we throw: ", len(outlier_Idx))
    print("- Remaining number of datapoints: ", x_all.shape[0])

# Save the data
pickle.dump(DatasSet_All, open(DataSetPath + "/data"+'.p', "wb"))
