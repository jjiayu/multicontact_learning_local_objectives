# Input Arguments
# 1: Working Direcory Path (e.g. .../Rubbles)
# 2: Local Obj Tracking RollOut Folder Name (i.e. NN_OriginalForm_....)
# 3: Ground Truth RollOut Folder Name (i.e. CleanTrainingSetRollOuts)
# 4: Folder to Store Unseen Configuration and Environments
# 5: Threshold to say two quantities are different

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


# --------------
# Set up for Directories
#   Define Working Directory
workingDirectory = sys.argv[1]
print("Working folder: ", workingDirectory)
#   Define Tracking RollOuts Path
TrackingExpPath = workingDirectory + "/" + sys.argv[2]
if not (os.path.isdir(TrackingExpPath)):
    raise Exception("Fault Tracking Exp Path")
else:
    print("Tracking Exp RollOut folder: ", TrackingExpPath)
#   Define Ground Truth RollOuts Path
GroundTruthPath = workingDirectory + "/" + sys.argv[3]
if not (os.path.isdir(GroundTruthPath)):
    raise Exception("Fault Ground Truth RollOut Path")
else:
    print("Ground Truth RollOut folder: ", GroundTruthPath)
#   Define and make folder to store unseen state
UnseenStateFolder = workingDirectory + "/" + sys.argv[4]
# NOTE: Report if we already has a folder with the same name, maybe we need to backup
if os.path.isdir(UnseenStateFolder):
    raise Exception(
        "Folder with the same name already exists, Backup the original folder and re-name the folder we want to use")
else:
    os.mkdir(UnseenStateFolder)
    print("Folder to Store Unseen States: ", UnseenStateFolder)

# ------------------------------------
#   Threshold to identify two quantities are different
diff_threshold = float(sys.argv[5])
print("Threshold for different state: ", diff_threshold)


# Give some time to show the working directory
time.sleep(7)


# ------------------------------------
# Define Group Prefix List
# GroupPrefix_List = ["Group1_",  "Group2_",  "Group3_",  "Group4_",  "Group5_",  "Group6_",  "Group7_",  "Group8_",  "Group9_",  "Group10_",
#                     "Group11_", "Group12_", "Group13_", "Group14_", "Group15_", "Group16_", "Group17_", "Group18_", "Group19_", "Group20_",
#                     "Group21_", "Group22_", "Group23_", "Group24_", "Group25_", "Group26_", "Group27_", "Group28_", "Group29_", "Group30_"]

# ------------------------------------
# Start Searching for Unseen States

#   Define Number of file proccessed
total_file_num = 0
unseen_state_file_num = 0

total_num_unseen_state = 0

#   Make a list as counter for diverging state at which round
divergingRoundCnt = [0]*50

#   Get File names in the Local Obj Tracking Exp Folder
TrackingExpfilenames = os.listdir(TrackingExpPath)

# Loop over all the tracking exp files
for TrackingExp_filename in TrackingExpfilenames:

    #   indicators of divergeing state within a round
    divergingRound_indicator = [0]*50

    if ".p" in TrackingExp_filename:  # a data file
        print("Process Tracking Exp File: ", TrackingExp_filename)

        total_file_num = total_file_num + 1  # Process one file

        # Make dir for this group if we need to
        UnseenStateStoragePath = UnseenStateFolder

        # Load Tracking Exp data
        with open(TrackingExpPath + '/' + TrackingExp_filename, 'rb') as f:
            TrackingExp_data = pickle.load(f)

        # Load Ground Truth data
        #   Ground Truth File name
        GroundTruthFileName_StartingIdx = TrackingExp_data["LocalObjSettings"]["GroundTruthTraj"].find(
            "/Group")
        GroundTruthFileName = TrackingExp_data["LocalObjSettings"][
            "GroundTruthTraj"][GroundTruthFileName_StartingIdx+1:]
        GroundTruth_file_path = GroundTruthPath + '/' + GroundTruthFileName
        with open(GroundTruth_file_path, 'rb') as f:
            GroundTruth_data = pickle.load(f)

        # ------------
        # Loop over each rounds to get Unseen states
        #   Clear the Unseen State Numbering
        UnseenStateCnt = 0
        #   Clear Unseen State Flag (True if there is at leason one unseen state, for stat purpose)
        UnseenStateFlag = False

        # NOTE: Skip Round 0 Failures that we dont have any savings
        for roundNum in range(1, len(TrackingExp_data["SingleOptResultSavings"])):

            # Get Opt Result for Local obj tracking and Ground Truth of current round
            TrackingExp_SingleOptRes = TrackingExp_data["SingleOptResultSavings"][roundNum]
            GroundTruth_SingleOptRes = GroundTruth_data["SingleOptResultSavings"][roundNum]

            TrackingExp_InitConfig, TrackingExp_TerminalConfig = getInitConfig_in_GlobalFrame_from_SingleOptResult(
                SingleOptRes=TrackingExp_SingleOptRes)
            GroundTruth_InitConfig, GroundTruth_TerminalConfig = getInitConfig_in_GlobalFrame_from_SingleOptResult(
                SingleOptRes=GroundTruth_SingleOptRes)

            # Build Difference between Init State Vectors for Tracking Exp and Ground Truth
            TrackingExp_State = np.array([TrackingExp_InitConfig["x_init"], TrackingExp_InitConfig["y_init"], TrackingExp_InitConfig["z_init"],
                                          TrackingExp_InitConfig["xdot_init"], TrackingExp_InitConfig[
                                              "ydot_init"], TrackingExp_InitConfig["zdot_init"],
                                          TrackingExp_InitConfig["Lx_init"], TrackingExp_InitConfig[
                                              "Ly_init"], TrackingExp_InitConfig["Lz_init"],
                                          TrackingExp_InitConfig["PLx_init"], TrackingExp_InitConfig[
                                              "PLy_init"], TrackingExp_InitConfig["PLz_init"],
                                          TrackingExp_InitConfig["PRx_init"], TrackingExp_InitConfig["PRy_init"], TrackingExp_InitConfig["PRz_init"]])

            GroundTruth_State = np.array([GroundTruth_InitConfig["x_init"], GroundTruth_InitConfig["y_init"], GroundTruth_InitConfig["z_init"],
                                          GroundTruth_InitConfig["xdot_init"], GroundTruth_InitConfig[
                                              "ydot_init"], GroundTruth_InitConfig["zdot_init"],
                                          GroundTruth_InitConfig["Lx_init"], GroundTruth_InitConfig[
                                              "Ly_init"], GroundTruth_InitConfig["Lz_init"],
                                          GroundTruth_InitConfig["PLx_init"], GroundTruth_InitConfig[
                                              "PLy_init"], GroundTruth_InitConfig["PLz_init"],
                                          GroundTruth_InitConfig["PRx_init"], GroundTruth_InitConfig["PRy_init"], GroundTruth_InitConfig["PRz_init"]])

            # Compute Error
            error_vec = np.linalg.norm(TrackingExp_State - GroundTruth_State)

            if error_vec > diff_threshold:

                total_num_unseen_state = total_num_unseen_state + 1

                divergingRound_indicator[roundNum] = divergingRound_indicator[roundNum] + 1

                UnseenStateFlag = True  # Found one Unseen state, turn the flat for future stat purposes

                # Save Unseen State (from local obj tracking exp) and Environment Model
                UnseenState = {}
                UnseenState["TrackingExpPath"] = TrackingExpPath + \
                    '/' + TrackingExp_filename
                UnseenState["Diverge_RoundNum"] = roundNum
                UnseenState["InitConfig"] = TrackingExp_InitConfig
                UnseenState["TerrainInfo"] = {"InitLeftSurfVertice":     TrackingExp_InitConfig["LeftInitSurf"],
                                              "InitLeftSurfTangentX":    TrackingExp_InitConfig["PL_init_TangentX"],
                                              "InitLeftSurfTangentY":    TrackingExp_InitConfig["PL_init_TangentY"],
                                              "InitLeftSurfNorm":        TrackingExp_InitConfig["PL_init_Norm"],
                                              "InitLeftSurfOrientation": TrackingExp_InitConfig["LeftInitSurfOrientation"],
                                              "InitRightSurfVertice":    TrackingExp_InitConfig["RightInitSurf"],
                                              "InitRightSurfTangentX":   TrackingExp_InitConfig["PR_init_TangentX"],
                                              "InitRightSurfTangentY":   TrackingExp_InitConfig["PR_init_TangentY"],
                                              "InitRightSurfNorm":       TrackingExp_InitConfig["PR_init_Norm"],
                                              "InitRightSurfOrientation": TrackingExp_InitConfig["RightInitSurfOrientation"],
                                              "ContactSurfsVertice": TrackingExp_data["TerrainInfo"]["ContactSurfsVertice"][roundNum:],
                                              "ContactSurfsHalfSpace": TrackingExp_data["TerrainInfo"]["ContactSurfsHalfSpace"][roundNum:],
                                              "ContactSurfsTypes": TrackingExp_data["TerrainInfo"]["ContactSurfsTypes"][roundNum:],
                                              "ContactSurfsNames": TrackingExp_data["TerrainInfo"]["ContactSurfsNames"][roundNum:],
                                              "ContactSurfsTangentX": TrackingExp_data["TerrainInfo"]["ContactSurfsTangentX"][roundNum:],
                                              "ContactSurfsTangentY": TrackingExp_data["TerrainInfo"]["ContactSurfsTangentY"][roundNum:],
                                              "ContactSurfsNorm": TrackingExp_data["TerrainInfo"]["ContactSurfsNorm"][roundNum:],
                                              "ContactSurfsOrientation": TrackingExp_data["TerrainInfo"]["ContactSurfsOrientation"][roundNum:],
                                              "AllPatchesVertices": [TrackingExp_InitConfig["LeftInitSurf"], TrackingExp_InitConfig["RightInitSurf"]] + TrackingExp_data["TerrainInfo"]["ContactSurfsVertice"][roundNum:]
                                              }
                UnseenState["TerrainSettings"] = TrackingExp_data["TerrainSettings"]

                # Update Unseen State Count
                UnseenStateCnt = UnseenStateCnt + 1

                # Save the Unseen State File
                pickle.dump(UnseenState, open(UnseenStateStoragePath + '/' +
                            TrackingExp_filename[:-2] + '_UnseenState_at_Round' + str(roundNum) + ".p", "wb"))  # Save Data

        if UnseenStateFlag == True:
            unseen_state_file_num = unseen_state_file_num + 1
            print("   - RollOut has Unseen State, all Saved")

        print(" ")

        # Count how many times of divergence firstly from a round
        if np.sum(np.array(divergingRound_indicator)) > 0.0:  # Note fail from the first step
            # ignore the first element to avoid failures at round 0
            firstDivergingRound = divergingRound_indicator.index(1)
            #print("Diverging Round Indicator: ", divergingRound_indicator)
            print("First Time Diverge from: ", firstDivergingRound)
            divergingRoundCnt[firstDivergingRound] = divergingRoundCnt[firstDivergingRound] + 1
            # print(divergingRoundCnt)


print("-------Summary--------")
print("Total number of Tracking Exp rollouts being processed: ", total_file_num)
print("Threshold for identifying different state: ", diff_threshold)
print("Total number of RollOuts that has unseen state: ", unseen_state_file_num)
print(" ")
print("Total Number of Unseen State Collected: ", total_num_unseen_state)
# print("Diverged Round Stats")
# for divergeRound in range(1,TrackingExp_data["Num_of_Rounds"]):
#     print("Diverge from Round ", str(divergeRound), ": ", str(divergingRoundCnt[divergeRound]), " times.")
