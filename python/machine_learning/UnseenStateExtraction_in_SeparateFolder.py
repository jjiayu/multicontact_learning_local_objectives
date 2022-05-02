# Input Arguments
# 1: Working Direcory Path (e.g. .../Rubbles)
# 2: Local Obj Tracking RollOut Folder Name (i.e. NN_OriginalForm_....)
# 3: Ground Truth RollOut Folder Name (i.e. CleanTrainingSetRollOuts)
# 4: Folder to Store Unseen Configuration and Environments
# 5: Threshold to say two quantities are different (unit: meters , i.e 0.005 = 5mm)
# 6: Get Unseen State from only Failed RollOuts or Success RollOuts: 1) Failed 2) All

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

# Collect the flat of getting Unseen State from what kind of RollOuts
getUnseenStatefrom = sys.argv[6]
print("Get Unseen State from :", getUnseenStatefrom, " RollOuts")

# ------------------------------------
#   Threshold to identify two quantities are different
diff_threshold = float(sys.argv[5])
print("Threshold for different state: ", diff_threshold)


# Give some time to show the working directory
time.sleep(7)


# ------------------------------------
# Define Group Prefix List
GroupPrefix_List = ["Group1_",  "Group2_",  "Group3_",  "Group4_",  "Group5_",  "Group6_",  "Group7_",  "Group8_",  "Group9_",  "Group10_",
                    "Group11_", "Group12_", "Group13_", "Group14_", "Group15_", "Group16_", "Group17_", "Group18_", "Group19_", "Group20_",
                    "Group21_", "Group22_", "Group23_", "Group24_", "Group25_", "Group26_", "Group27_", "Group28_", "Group29_", "Group30_"]

# ------------------------------------
# Start Searching for Unseen States

#   Define Number of file proccessed
total_file_num = 0
unseen_state_file_num = 0

#   Make a list as counter for diverging state at which round
divergingRoundCnt = [0]*50

#   Get File names in the Local Obj Tracking Exp Folder
TrackingExpfilenames = os.listdir(TrackingExpPath)

# Loop over all the tracking exp files
for TrackingExp_filename in TrackingExpfilenames:

    # Clear Important Flags
    #   Failed or success rollout indicator
    FailedRound = False
    #   indicators of divergeing state within a round
    divergingRound_indicator = [0]*50

    if ".p" in TrackingExp_filename:  # a data file
        print("Process Tracking Exp File: ", TrackingExp_filename)

        total_file_num = total_file_num + 1  # Process one file

        # Check Which Group prefix
        for groupfix_temp in GroupPrefix_List:
            if groupfix_temp in TrackingExp_filename:
                group_prefix = groupfix_temp[:-1]

        # Make dir for this group if we need to
        UnseenStateStoragePath = UnseenStateFolder + '/' + group_prefix
        if not (os.path.isdir(UnseenStateStoragePath)):
            os.mkdir(UnseenStateStoragePath)
        #print("   - Should store Unseen State in (Prefix: ", group_prefix, "): ", UnseenStateStoragePath)

        # Load Tracking Exp data
        with open(TrackingExpPath + '/' + TrackingExp_filename, 'rb') as f:
            TrackingExp_data = pickle.load(f)

        # Load Ground Truth data
        #   Ground Truth File name
        GroundTruth_file_path = TrackingExp_data["LocalObjSettings"]["GroundTruthTraj"]
        with open(GroundTruth_file_path, 'rb') as f:
            GroundTruth_data = pickle.load(f)

        # Check if the current round is a failued round or success round
        if len(TrackingExp_data["SingleOptResultSavings"]) < TrackingExp_data["Num_of_Rounds"]:
            FailedRound = True
        else:
            FailedRound = False

        if FailedRound == False and getUnseenStatefrom == "Failed":
            #print("   We want Unseen State from ", getUnseenStatefrom, " Rounds, but this round is Success, so we then jump current loop")
            #print(" ")
            continue

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
            InitState_Diff = np.array([TrackingExp_InitConfig["x_init"]-GroundTruth_InitConfig["x_init"],
                                       TrackingExp_InitConfig["y_init"] -
                                       GroundTruth_InitConfig["y_init"],
                                       TrackingExp_InitConfig["z_init"] -
                                       GroundTruth_InitConfig["z_init"],
                                       TrackingExp_InitConfig["xdot_init"] -
                                       GroundTruth_InitConfig["xdot_init"],
                                       TrackingExp_InitConfig["ydot_init"] -
                                       GroundTruth_InitConfig["ydot_init"],
                                       TrackingExp_InitConfig["zdot_init"] -
                                       GroundTruth_InitConfig["zdot_init"],
                                       TrackingExp_InitConfig["Lx_init"] -
                                       GroundTruth_InitConfig["Lx_init"],
                                       TrackingExp_InitConfig["Ly_init"] -
                                       GroundTruth_InitConfig["Ly_init"],
                                       TrackingExp_InitConfig["Lz_init"] -
                                       GroundTruth_InitConfig["Lz_init"],
                                       TrackingExp_InitConfig["PLx_init"] -
                                       GroundTruth_InitConfig["PLx_init"],
                                       TrackingExp_InitConfig["PLy_init"] -
                                       GroundTruth_InitConfig["PLy_init"],
                                       TrackingExp_InitConfig["PLz_init"] -
                                       GroundTruth_InitConfig["PLz_init"],
                                       TrackingExp_InitConfig["PRx_init"] -
                                       GroundTruth_InitConfig["PRx_init"],
                                       TrackingExp_InitConfig["PRy_init"] -
                                       GroundTruth_InitConfig["PRy_init"],
                                       TrackingExp_InitConfig["PRz_init"]-GroundTruth_InitConfig["PRz_init"]])
            InitState_Diff = np.absolute(InitState_Diff)

            # Compare with respect to threshold
            Diff_Boolean_vec = InitState_Diff > diff_threshold

            # Store Unseen State and Environment Model, if there is one axis is larger than the threshold
            if Diff_Boolean_vec.sum() > 0:
                #print("Found Unseen State at Round: ", roundNum)
                #print("Diffs: ", InitState_Diff)
                # Update counter as well
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
                if getUnseenStatefrom == "Failed":
                    if FailedRound == True:
                        pickle.dump(UnseenState, open(UnseenStateStoragePath + '/' +
                                    TrackingExp_filename[:-2] + '_UnseenState' + str(UnseenStateCnt) + ".p", "wb"))  # Save Data
                elif getUnseenStatefrom == "All":
                    pickle.dump(UnseenState, open(UnseenStateStoragePath + '/' +
                                TrackingExp_filename[:-2] + '_UnseenState' + str(UnseenStateCnt) + ".p", "wb"))  # Save Data
                else:
                    raise Exception(
                        "Unknow Flag of getting unseen state from what kind of rollouts")

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
print("Diverged Round Stats")
for divergeRound in range(1, TrackingExp_data["Num_of_Rounds"]):
    print("Diverge from Round ", str(divergeRound), ": ",
          str(divergingRoundCnt[divergeRound]), " times.")
