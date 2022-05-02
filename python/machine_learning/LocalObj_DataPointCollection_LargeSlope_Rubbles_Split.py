# Input Arguments:
# 1: Working Directory (The root folder of the work space)
# 2: RollOutfolder name
# 3: DataSet Folder name

# Import Packages
import numpy as np
import os
import pickle
from multicontact_learning_local_objectives.python.machine_learning.ml_utils import *
from multicontact_learning_local_objectives.python.terrain_create import *
import matplotlib.pyplot as plt  # Matplotlib
import time
import shutil
import sys

# Get Working Directory and dataset type
workingDirectory = sys.argv[1]
RollOutFoldername = sys.argv[2]
print("Working Folder: \n", workingDirectory)

# Define Large Slope Deicison Boundary, NOTE: we generate additional separete dataset for large slope and small inclinations
LargeSlopeDecisionBoundary = 17.0

# ------------------------
# Definr RollOut Path
rolloutPath = workingDirectory + '/' + \
    RollOutFoldername  # "/CleanTrainingSetRollOuts/"

# Path to store dataset
DataSetPath = workingDirectory + "/" + sys.argv[3] + "/"
# make the folder if we dont have one
if not (os.path.isdir(DataSetPath)):
    os.mkdir(DataSetPath)

# Give some time to show the working directory
time.sleep(7)

# -----------------------
# Define Frame Transformation and DataPoint Representation
# Shift world frame type: StanceFoot, (Abandon) InitCoM,  (Abandon) InitSurfBorder, None
Shift_World_Frame_Type = "StanceFoot"
print("Shift World Frame to:\n", Shift_World_Frame_Type)
# Contact Location Representation Type 1) 3DPoints 2) ConvexCombination 3) FollowRectangelBorder
Contact_Representation_Type = "FollowRectangelBorder"
print("Contact Location Representation Type:\n ", Contact_Representation_Type)
# Scaling Vectors? 1 menas no scaling (Now scaling does not make much diff)
ScaleFactor = 1
print("Scaling Factor of all quantities (Except Left and Right Swing Flag):\n", ScaleFactor)

# ---------------
# Start Collecting Data Points
# get all the file names
filenames = os.listdir(rolloutPath)

# Initialize Dataset
x_all = []
y_all = []
x_largeslope = []
y_largeslope = []
x_rubbles = []
y_rubbles = []

#fig, ax = plt.subplots()

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
        print(data["TerrainSettings"])

        if len(data["SingleOptResultSavings"]) == data["Num_of_Rounds"]:

            print("Success Computation")
            success_file_num = success_file_num + 1

            for roundNum in range(data["Num_of_Rounds"]):
                print("   Process Round: ", roundNum)
                # Get Single Optimization Result of current result
                singleOptResult = data["SingleOptResultSavings"][roundNum]

                # Get data point
                xtemp, ytemp = getDataPoints(SingleOptRes=singleOptResult, Shift_World_Frame=Shift_World_Frame_Type,
                                             ContactRepresentationType=Contact_Representation_Type, VectorScaleFactor=ScaleFactor)  # InitCoM; InitSurfBorder

                # -----------
                # Check if contains large slope, and then store into different arrays
                LargeSlopeFlag = 0

                #   Check left init patch
                leftInitRotationAngle = np.absolute(
                    getTerrainRotationAngle(singleOptResult["LeftInitSurf"]))
                if leftInitRotationAngle >= LargeSlopeDecisionBoundary:
                    LargeSlopeFlag = 1

                #   Check right init patch
                rightInitRotationAngle = np.absolute(
                    getTerrainRotationAngle(singleOptResult["RightInitSurf"]))
                if rightInitRotationAngle >= LargeSlopeDecisionBoundary:
                    LargeSlopeFlag = 1

                for surfIdx in range(len(singleOptResult["ContactSurfs"])):
                    # if surfIdx == 0 or surfIdx == 1:
                    surftemp = singleOptResult["ContactSurfs"][surfIdx]
                    surftempRotationAngle = np.absolute(
                        getTerrainRotationAngle(surftemp))
                    if surftempRotationAngle >= LargeSlopeDecisionBoundary:
                        LargeSlopeFlag = 1

                # Put datapoint into different dataset
                if LargeSlopeFlag == 1:
                    print("Found Large Slope, Put into Large Slope DataSet")
                    x_largeslope.append(xtemp)
                    y_largeslope.append(ytemp)
                elif LargeSlopeFlag == 0:
                    print("No Large Slope Found, Put into Rubbles DataSet")
                    x_rubbles.append(xtemp)
                    y_rubbles.append(ytemp)
                else:
                    raise Exception("Unknow Slope type")

                print("Put into the complete dataset as well")
                x_all.append(xtemp)
                y_all.append(ytemp)

        else:
            print("Failed Computation")

# make data points to become numpy array
x_all = np.array(x_all)
y_all = np.array(y_all)
x_largeslope = np.array(x_largeslope)
y_largeslope = np.array(y_largeslope)
x_rubbles = np.array(x_rubbles)
y_rubbles = np.array(y_rubbles)

# Get NumberofPreviewSteps
NumPreviewSteps = len(data["SingleOptResultSavings"][0]["ContactSurfs"])

# Save DataPoints and Settings
DatasSet_All = {"input":  x_all,
                "output": y_all,
                "Shift_World_Frame_Type": Shift_World_Frame_Type,
                "Contact_Representation_Type": Contact_Representation_Type,
                "VectorScaleFactor": ScaleFactor,
                "NumPreviewSteps": NumPreviewSteps}

DataSet_LargeSlope = {"input":  x_largeslope,
                      "output": y_largeslope,
                      "Shift_World_Frame_Type": Shift_World_Frame_Type,
                      "Contact_Representation_Type": Contact_Representation_Type,
                      "VectorScaleFactor": ScaleFactor,
                      "NumPreviewSteps": NumPreviewSteps}

DataSet_Rubbles = {"input":  x_rubbles,
                   "output": y_rubbles,
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

print(" ")
print("Summary: ")
print("- Total Number of Rollouts (files): ", total_file_num)
print("- Total Number of Success Rollouts (files): ", success_file_num)

# Save the data
pickle.dump(DatasSet_All, open(DataSetPath + "/data"+'.p', "wb"))
pickle.dump(DataSet_LargeSlope, open(
    DataSetPath + "/data_largeslope"+'.p', "wb"))
pickle.dump(DataSet_Rubbles, open(DataSetPath + "/data_rubbles"+'.p', "wb"))
