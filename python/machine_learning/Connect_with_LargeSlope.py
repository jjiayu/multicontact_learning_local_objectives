import numpy as np
# Need get Init Config from global frame Function
from multicontact_learning_local_objectives.python.rhp_plan.rhp_utils import *
import os
import pickle
from multicontact_learning_local_objectives.python.machine_learning.ml_utils import *
import matplotlib.pyplot as plt  # Matplotlib
import time
import shutil
import sys
from multicontact_learning_local_objectives.python.rhp_plan.rhp_utils import *
from multicontact_learning_local_objectives.python.terrain_create import *
import multicontact_learning_local_objectives.python.visualization as viz

#
NN_track_largslope_filename = "Group1_temp1642001313.0230558.p"

# --------------
# Set up for Directories
#   Define Working Directory
workingDirectory = sys.argv[1]
print("Working folder: ", workingDirectory)

#   Define the the folder where we get the terrain the model (CleanTraining/Vallidation etc.)
TerrainModel_Folder_Name = sys.argv[2]
TerrainModel_Folder_Path = workingDirectory + '/' + TerrainModel_Folder_Name
print("The folder where stores the Terrain/ground-truth local obj for tracking the first few rounds: ",
      TerrainModel_Folder_Name)

#   Define the folder where we get the large slope tracking result
LargeSlope_TrackingFolder_Name = sys.argv[3]
print("The folder where stores the Terrain/ground-truth local obj for tracking the first few rounds: ",
      LargeSlope_TrackingFolder_Name)

#   Define the folder where stroe the rubbles result
Rubbles_Folder_Name = sys.argv[4]
print("The folder where stores the Terrain/ground-truth local obj for tracking the first few rounds: ", Rubbles_Folder_Name)

# ------------------
# Now We start Processing

# Load the Large slope Tracking file
NN_track_largslope_file_path = workingDirectory + '/' + \
    LargeSlope_TrackingFolder_Name + '/' + NN_track_largslope_filename
with open(NN_track_largslope_file_path, 'rb') as f:
    NN_track_largslope_data = pickle.load(f)

print(NN_track_largslope_data["TerrainModelPath"])

# Then load the Terrain File
TerrainModelFileName_Idx = NN_track_largslope_data["TerrainModelPath"].find(
    "/Group")
TerrainModelFilePath = TerrainModel_Folder_Path + \
    NN_track_largslope_data["TerrainModelPath"][TerrainModelFileName_Idx:]
print(TerrainModelFilePath)

with open(TerrainModelFilePath, 'rb') as f:
    TerrainModelFile_data = pickle.load(f)

print(TerrainModelFile_data["TerrainModelPath"])

# Then load the terrain file where we get the partial large slope terrain
PartialTerrainFileName_Idx = TerrainModelFile_data["TerrainModelPath"].find(
    "LargeSlopeTerrain")
PartialTerrainFilePath = workingDirectory + '/' + \
    TerrainModelFile_data["TerrainModelPath"][PartialTerrainFileName_Idx:]

with open(PartialTerrainFilePath, 'rb') as f:
    PartialTerrain_data = pickle.load(f)

print(PartialTerrain_data["Rubbles_RollOut_filepath"])
print("----------------")
StepIdx_whereLargeSlopeTerrainStarts = PartialTerrain_data["StepIdx_of_StartofTerrain"]
print(PartialTerrain_data["StepIdx_of_StartofTerrain"])
print("----------------")
# print(PartialTerrain_data.keys())

# Then we get Rubbles RollOut data
Rubbles_FileName_Idx = PartialTerrain_data["Rubbles_RollOut_filepath"].find(
    "/Group")
Rubbles_FilePath = workingDirectory + '/' + Rubbles_Folder_Name + '/' + \
    PartialTerrain_data["Rubbles_RollOut_filepath"][Rubbles_FileName_Idx:]

with open(Rubbles_FilePath, 'rb') as f:
    Rubbles_data = pickle.load(f)

ConnectedData = copy.deepcopy(Rubbles_data)

ConnectedData["TerrainModel"] = Rubbles_data["TerrainModel"][0:
                                                             StepIdx_whereLargeSlopeTerrainStarts] + NN_track_largslope_data["TerrainModel"]
ConnectedData["CasadiParameters"] = Rubbles_data["CasadiParameters"][0:
                                                                     StepIdx_whereLargeSlopeTerrainStarts] + NN_track_largslope_data["CasadiParameters"]
ConnectedData["SingleOptResultSavings"] = Rubbles_data["SingleOptResultSavings"][0:
                                                                                 StepIdx_whereLargeSlopeTerrainStarts] + NN_track_largslope_data["SingleOptResultSavings"]
ConnectedData["Trajectory_of_All_Rounds"] = Rubbles_data["Trajectory_of_All_Rounds"][0:
                                                                                     StepIdx_whereLargeSlopeTerrainStarts] + NN_track_largslope_data["Trajectory_of_All_Rounds"]
ConnectedData["Num_of_Rounds"] = len(Rubbles_data["SingleOptResultSavings"]
                                     [0:StepIdx_whereLargeSlopeTerrainStarts]) + len(NN_track_largslope_data["SingleOptResultSavings"])

pickle.dump(ConnectedData, open(
    workingDirectory + '/' + 'testtest' + ".p", "wb"))
