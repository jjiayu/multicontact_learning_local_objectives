# Input Arguments:
# 1: Working Directory (The root folder of the work space)
# 2: RollOutfolder name
# 3: DataSet Folder name
# 4: Pre Processing Mode: 1) "OriginalForm" 2) "Standarization" 3) "MaxAbs" (lies in [-1, 1])
# 5: Scale Factor
# 6: Remove Outlier or not 1) Remove3SigmaOutlier 2)KeepOutlier 3) RemovebyClip 4)RemoveMultiGaussainOutlier
# 7: Ground Truth RollOut Folder Name
# 8: Unseen State Folder Name
# 9: Dist Threshold

# python3 LocalObj_DataPointCollection_VarLength.py /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/LargeSlope_TimeTrack_Angle_17_26 RollOuts_8Steps_from_3StepbeforeFail_TrackTrainingInit_Aug_3Time_EarlyStopping TrainingAug4Time_3StepBeforeFail_TrackTrainingInit_Aug_3Time_EarlyStopping OriginalForm 1.0 KeepOutlier CleanTrainingRollOuts_Init Unseen_3StepbeforeFail_TrackingTrainingInit_Aug_3Time_EarlyStopping 0.06

# python3 LocalObj_DataPointCollection_VarLength.py /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/Rubbles_TimeTrack RollOuts_10Steps_from_1StepbeforeFail_TrackTrainingAll_InitSet TrainingAug1Time_1StepBeforeFail_TrackTrainingInit OriginalForm 1.0 RemovebyClip CleanTrainingSetRollOuts_All Unseen_1StepbeforeFail_TrackingTrainingAll_InitSet 0.06

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

# Define Ground Truth Roll Out Folder Path
groundtruth_rollout_Path = workingDirectory + '/' + sys.argv[7]

# Define Unseen State Folder Name
unseenstate_Path = workingDirectory + '/' + sys.argv[8]

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

# #Get start and terminal round number
# startroundNum = int(sys.argv[6])
# terminalroundNum = int(sys.argv[7])
# print("Start Collection from Round Number: ", startroundNum)
# print("Stop Collection at Round Number:", terminalroundNum)

# Dealing with Outlier
OutlierFlag = sys.argv[6]

# dist threshold
dist_threshold = float(sys.argv[9])

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


# History of Recovery Traj Length
recovery_length_record = []

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

            # Read Ground Truth Traj
            UnseenStateFilePath = data["TerrainModelPath"]
            idxTemp = UnseenStateFilePath.find("Group")
            UnseenStateFilePath = unseenstate_Path + \
                "/" + UnseenStateFilePath[idxTemp:]
            UnseenState_data = pickle.load(open(UnseenStateFilePath, "rb"))

            GroundTruthFilePath = UnseenState_data["GroundTruthFile_Path"]
            idxTemp = GroundTruthFilePath.find("Group")
            GroundTruthFilePath = groundtruth_rollout_Path + \
                "/" + GroundTruthFilePath[idxTemp:]
            GroundTruth_data = pickle.load(open(GroundTruthFilePath, "rb"))

            # Get Rounds of Finding Unseen State
            UnseenStateRoundIdx = UnseenState_data["UnseenRoundIdx"]

            # distance vector
            dist_vec = []

            # Build the dist vec
            # range(data["Num_of_Rounds"]):
            for roundNum in range(len(data["SingleOptResultSavings"])):
                #print("   Process Round: ", roundNum)
                # Get Single Optimization Result of current result
                singleOptResult = data["SingleOptResultSavings"][roundNum]
                GroundTruth_Opt_Result = GroundTruth_data["SingleOptResultSavings"]

                # Get data point
                xtemp, ytemp = getDataPoints(SingleOptRes=singleOptResult, Shift_World_Frame=Shift_World_Frame_Type,
                                             ContactRepresentationType=Contact_Representation_Type, VectorScaleFactor=ScaleFactor)  # InitCoM; InitSurfBorder

                # Get Ground Truth Data Points
                if UnseenStateRoundIdx + roundNum > len(GroundTruth_data["SingleOptResultSavings"]) - 1:
                    break  # Out of Ground Truth Traj Length
                elif UnseenStateRoundIdx + roundNum <= len(GroundTruth_data["SingleOptResultSavings"]) - 1:
                    x_ground_truth, y_ground_truth = getDataPoints(SingleOptRes=GroundTruth_Opt_Result[UnseenStateRoundIdx + roundNum], Shift_World_Frame=Shift_World_Frame_Type,
                                                                   ContactRepresentationType=Contact_Representation_Type, VectorScaleFactor=ScaleFactor)  # InitCoM; InitSurfBorder
                else:
                    raise Exception("Unknown Problem")

                # Compute dist
                dist_vec.append(np.linalg.norm(xtemp - x_ground_truth))

            # Compare Against Threshold
            largeError_flag = list(np.array(dist_vec) > dist_threshold)
            # print(np.round(dist_vec,3))
            # print(largeError_flag)

            # Collect Data Points
            # Decide Length of Recovery Traj
            # Did not find small error steps until the end (most probably last few rounds)
            if np.sum(largeError_flag) == len(largeError_flag):
                RecoveryTraj_Length = len(largeError_flag)
            # False starts from very beginning, need to add at least one round
            elif largeError_flag[0] == False:
                RecoveryTraj_Length = 1
            else:
                RecoveryTraj_Length = largeError_flag.index(False) + 1 - 1
            print("Recovery Traj Length Start from Round: ",
                  UnseenStateRoundIdx, "with Length: ", RecoveryTraj_Length)
            # Record the value
            recovery_length_record.append(RecoveryTraj_Length)

            for RoundNum in range(RecoveryTraj_Length):
                singleOptResult = data["SingleOptResultSavings"][RoundNum]
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
num_datapoint_we_should = x_all.shape[0]

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
                "NumPreviewSteps": NumPreviewSteps,
                "Dist_Threshold": dist_threshold}

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
print("- Total Number of DataPoint we should collect: ", num_datapoint_we_should)
if OutlierFlag == "Remove3SigmaOutlier" or OutlierFlag == "RemovebyClip":
    print("- Outliersthat we throw: ", len(outlier_Idx))
    print("- Remaining number of datapoints: ", x_all.shape[0])

# Plot Histogram
plt.hist(recovery_length_record, bins=20, density=False)
plt.show()

# Save the data
pickle.dump(DatasSet_All, open(DataSetPath + "/data"+'.p', "wb"))
