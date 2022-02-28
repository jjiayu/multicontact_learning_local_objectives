#Input Arguments
#1: Working Direcory Path (e.g. .../Rubbles)
#2: Name of the Rubbles Rollout Folder
#3: Name of Folder to Store these large slope env
#4: From which Step shall we start to create the large slope env? (count from 0) 
#       Note: the swing foot land on this idx, we just take initial patch of this step
#5: How many steps later shall we put a large slope? (count from 0); 
#      if we want the first round has large slope in the horizon, we should have large slope index = Num Lookahead - 1
#6: Min rotation angle for the large slope (17 degrees)
#7: Max rotation angle for the large slope (26 degrees)
#8: Rotate Axis: None, "X_positive", "X_negative", "Y_positive", "Y_negative"

#python3 machine_learning/Create_LargeSlopeTerrain_from_Rollouts.py /media/jiayu/Seagate/LargeSlope Rubbles_CleanTrainingSetRollOuts_All Largeslopetest 13 5 17 26

#python3 Create_LargeSlopeTerrain_from_Rollouts.py /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/LargeSlope_Angle_21_26 Rubbles_CleanTrainingSetRollOuts_All LargeSlope_Start12_Large_5_Angle21_26 12 5 21 26

#python3 Create_LargeSlopeTerrain_from_Rollouts.py /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/LargeSlope_Angle_17_21 Rubbles_CleanTrainingSetRollOuts_All LargeSlope_Start12_Large_5_Angle17_21 12 5 17 21

#python3 Create_LargeSlopeTerrain_from_Rollouts.py /afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/datastorage/LargeSlope_Angle_22_X_negative Rubbles_CleanTrainingSetRollOuts_All LargeSlopeSetup_Start12_Large5_X_negative_Angle22 12 5 22 22 X_negative

import numpy as np
from multicontact_learning_local_objectives.python.rhp_plan.rhp_utils import * #Need get Init Config from global frame Function 
import os
import pickle
from multicontact_learning_local_objectives.python.machine_learning.ml_utils import *
import matplotlib.pyplot as plt #Matplotlib
import time
import shutil
import sys
from multicontact_learning_local_objectives.python.rhp_plan.rhp_utils import *
from multicontact_learning_local_objectives.python.terrain_create import *


#--------------
#Set up for Directories
#   Define Working Directory
workingDirectory = sys.argv[1]
print("Working folder: ", workingDirectory)

#   Define Rubbles NLP Rollout path
Rubbles_RollOut_Path = workingDirectory + "/" + sys.argv[2]
if not (os.path.isdir(Rubbles_RollOut_Path)):
    raise Exception("Fault Tracking Exp Path")
else:
    print("Tracking Exp RollOut folder: ", Rubbles_RollOut_Path)

#   Define and make folder to store large slope cases
LargeSlopeFolder = workingDirectory + "/" + sys.argv[3]
if os.path.isdir(LargeSlopeFolder): #NOTE: Report if we already has a folder with the same name, maybe we need to backup
    raise Exception("Folder with the same name already exists, Backup the original folder and re-name the folder we want to use")
else:
    os.mkdir(LargeSlopeFolder)
    print("Folder to Store Unseen States: ", LargeSlopeFolder)

#Step index from which we start our terrain
StepIdx_for_getting_StartofTerrain = int(sys.argv[4])
print("Our Terrain start from Step: ", StepIdx_for_getting_StartofTerrain, "Note: We land on this step, but the start of the terrain is actually the initial patches of this step")

#Large Slope Modification Idx
LargeSlopPatchIdx_relative_to_theFirstStep = int(sys.argv[5])
print("Index of the Large Slope Patch (at which step we have the large slope?): ", LargeSlopPatchIdx_relative_to_theFirstStep)

#Min rotation angle for the large slope
Min_LargeSlope_Angle = float(sys.argv[6])/180*np.pi
Max_LargeSlope_Angle = float(sys.argv[7])/180*np.pi

print("Min Large Slope Angle (in degrees): ",sys.argv[6])
print("Max Large Slope Angle (in degrees): ",sys.argv[7])

print("Min Large Slope Angle (in radius): ",Min_LargeSlope_Angle)
print("Max Large Slope Angle (in radius): ",Max_LargeSlope_Angle)


rotationType = sys.argv[8]
if rotationType == "None":
    rotationType = None

#Give some time to show the working directory
time.sleep(7)

#------------------------------------
#Start Searching for Unseen States

#   Define Number of file proccessed
total_file_num = 0

#   Get File names in the Local Obj Tracking Exp Folder
Rubbles_RollOut_filenames = os.listdir(Rubbles_RollOut_Path)

#Loop over all the tracking exp files
for Rubbles_Rollout_filename in Rubbles_RollOut_filenames:

    if ".p" in Rubbles_Rollout_filename:#a data file
        
        print("Process Rubble Rollout File (Computed from full NLP): ",Rubbles_Rollout_filename)
        total_file_num = total_file_num + 1 #Process one file
        
        #Load Tracking Exp data
        with open(Rubbles_RollOut_Path + '/' +Rubbles_Rollout_filename, 'rb') as f:
            Rubbles_RollOut_data= pickle.load(f)
        
        #Load Ground Truth data
        #   Ground Truth File name
        # GroundTruth_file_path = TrackingExp_data["LocalObjSettings"]["GroundTruthTraj"]
        # with open(GroundTruth_file_path, 'rb') as f:
        #     GroundTruth_data= pickle.load(f)

        #Check if the current round is a failued round or success round
        if len(Rubbles_RollOut_data["SingleOptResultSavings"]) == Rubbles_RollOut_data["Num_of_Rounds"]:
            #This is a successful rollout

            roundIdx_of_interst = StepIdx_for_getting_StartofTerrain

            SingleOptResult_for_StartofTerrain = Rubbles_RollOut_data["SingleOptResultSavings"][roundIdx_of_interst]

            #Get Init and Terminal Config of the plan
            Rubbles_RollOut_InitConfig, Rubbles_RollOut_TerminalConfig = getInitConfig_in_GlobalFrame_from_SingleOptResult(SingleOptRes=SingleOptResult_for_StartofTerrain)

            #Save Unseen State (from local obj tracking exp) and Environment Model
            LargeSlopeSetup = {}
            LargeSlopeSetup["Rubbles_RollOut_filepath"] = Rubbles_RollOut_Path + '/' + Rubbles_Rollout_filename
            LargeSlopeSetup["StepIdx_of_StartofTerrain"]=StepIdx_for_getting_StartofTerrain
            LargeSlopeSetup["LargeSlope_Idx"]=LargeSlopPatchIdx_relative_to_theFirstStep
            LargeSlopeSetup["Min_LargeSlope_Angle"] = Min_LargeSlope_Angle
            LargeSlopeSetup["Max_LargeSlope_Angle"] = Max_LargeSlope_Angle

            LargeSlopeSetup["InitConfig"] = Rubbles_RollOut_InitConfig

            #Modify Terrain Model
            TerrainModel = {"InitLeftSurfVertice":     Rubbles_RollOut_InitConfig["LeftInitSurf"],
                            "InitLeftSurfTangentX":    Rubbles_RollOut_InitConfig["PL_init_TangentX"],
                            "InitLeftSurfTangentY":    Rubbles_RollOut_InitConfig["PL_init_TangentY"],
                            "InitLeftSurfNorm":        Rubbles_RollOut_InitConfig["PL_init_Norm"],
                            "InitLeftSurfOrientation": Rubbles_RollOut_InitConfig["LeftInitSurfOrientation"],
                            "InitRightSurfVertice":    Rubbles_RollOut_InitConfig["RightInitSurf"],
                            "InitRightSurfTangentX":   Rubbles_RollOut_InitConfig["PR_init_TangentX"],
                            "InitRightSurfTangentY":   Rubbles_RollOut_InitConfig["PR_init_TangentY"],
                            "InitRightSurfNorm":       Rubbles_RollOut_InitConfig["PR_init_Norm"],
                            "InitRightSurfOrientation":Rubbles_RollOut_InitConfig["RightInitSurfOrientation"],
                            "ContactSurfsVertice":Rubbles_RollOut_data["TerrainInfo"]["ContactSurfsVertice"][roundIdx_of_interst:],
                            "ContactSurfsHalfSpace":Rubbles_RollOut_data["TerrainInfo"]["ContactSurfsHalfSpace"][roundIdx_of_interst:],
                            "ContactSurfsTypes": Rubbles_RollOut_data["TerrainInfo"]["ContactSurfsTypes"][roundIdx_of_interst:],
                            "ContactSurfsNames":Rubbles_RollOut_data["TerrainInfo"]["ContactSurfsNames"][roundIdx_of_interst:],
                            "ContactSurfsTangentX":Rubbles_RollOut_data["TerrainInfo"]["ContactSurfsTangentX"][roundIdx_of_interst:],
                            "ContactSurfsTangentY":Rubbles_RollOut_data["TerrainInfo"]["ContactSurfsTangentY"][roundIdx_of_interst:],
                            "ContactSurfsNorm":Rubbles_RollOut_data["TerrainInfo"]["ContactSurfsNorm"][roundIdx_of_interst:],
                            "ContactSurfsOrientation": Rubbles_RollOut_data["TerrainInfo"]["ContactSurfsOrientation"][roundIdx_of_interst:],
                            "AllPatchesVertices": [Rubbles_RollOut_InitConfig["LeftInitSurf"], Rubbles_RollOut_InitConfig["RightInitSurf"]] + Rubbles_RollOut_data["TerrainInfo"]["ContactSurfsVertice"][roundIdx_of_interst:]
                        }

            LargeSlopeSetup["TerrainInfo"], useful_Info = terrain_modify(TerrainModel = TerrainModel, PatchIdx_for_Mod = LargeSlopPatchIdx_relative_to_theFirstStep, 
                                                            ModificationMode = "rotate", 
                                                            RotateAxis = rotationType, RotateAngle = None, min_theta = Min_LargeSlope_Angle, max_theta = Max_LargeSlope_Angle) #Large slope ranging from 17 degrees to 26 degrees
            
            LargeSlopeSetup["TerrainSettings"] = Rubbles_RollOut_data["TerrainSettings"]

            LargeSlopeSetup["Rotation_Angle"] = useful_Info["rotation_angle"]

            #Save Files
            pickle.dump(LargeSlopeSetup, open(LargeSlopeFolder + '/' + Rubbles_Rollout_filename[:-2] + '_start_' + str(StepIdx_for_getting_StartofTerrain) + "_large_" + str(LargeSlopPatchIdx_relative_to_theFirstStep) + "_" + rotationType + "_angle_" + str(np.round(useful_Info["rotation_angle"]/np.pi*180,1)) + ".p", "wb"))    #Save Data
            print(" ")

print("-------Summary--------")
print("Total number of Tracking Exp rollouts being processed: ", total_file_num)
print(" ")