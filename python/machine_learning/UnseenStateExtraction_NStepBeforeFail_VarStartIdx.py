#Input Arguments
#1: Working Direcory Path (e.g. .../Rubbles)
#2: Local Obj Tracking RollOut Folder Name (i.e. NN_OriginalForm_....)
#3: Ground Truth RollOut Folder Name (i.e. CleanTrainingSetRollOuts)
#4: Folder to Store Unseen Configuration and Environments
#5: How many steps before failed tracking? (Count from 1)
#6: Which round Idx should we start the trace back (start idx of the traj of large slope) (i.e. 2)

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

#--------------
#Set up for Directories
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
if os.path.isdir(UnseenStateFolder): #NOTE: Report if we already has a folder with the same name, maybe we need to backup
    raise Exception("Folder with the same name already exists, Backup the original folder and re-name the folder we want to use")
else:
    os.mkdir(UnseenStateFolder)
    print("Folder to Store Unseen States: ", UnseenStateFolder)

#Get the index of how many steps before the failed round
StepIndexbeforeFail = int(sys.argv[5])
print("We collect the ", str(StepIndexbeforeFail), "-th step before the failed step")

#Trace back round Idx
StartRoundIdx_TraceBack = int(sys.argv[6])
print("We stop track back from Round Idx: ", StartRoundIdx_TraceBack)

#Give some time to show the working directory
time.sleep(7)

#------------------------------------
#Start Searching for Unseen States

#   Define Number of file proccessed
total_file_num = 0
failed_file_num = 0

#   Get File names in the Local Obj Tracking Exp Folder
TrackingExpfilenames = os.listdir(TrackingExpPath)


roundIdx_of_Interest_List = []
#Loop over all the tracking exp files
for TrackingExp_filename in TrackingExpfilenames:

    if ".p" in TrackingExp_filename:#a data file
        
        print("Process Tracking Exp File: ",TrackingExp_filename)
        total_file_num = total_file_num + 1 #Process one file
        
        #Load Tracking Exp data
        with open(TrackingExpPath + '/' +TrackingExp_filename, 'rb') as f:
            TrackingExp_data= pickle.load(f)
        
        #Load Ground Truth data
        #   Ground Truth File name
        # GroundTruth_file_path = TrackingExp_data["LocalObjSettings"]["GroundTruthTraj"]
        # with open(GroundTruth_file_path, 'rb') as f:
        #     GroundTruth_data= pickle.load(f)

        #Check if the current round is a failued round or success round
        if len(TrackingExp_data["SingleOptResultSavings"]) < TrackingExp_data["Num_of_Rounds"] and len(TrackingExp_data["SingleOptResultSavings"]) >= StepIndexbeforeFail: #also need to filter out the cases where the traj lenght is smaller than the expected before fail index
            
            roundIdx_of_interst = len(TrackingExp_data["SingleOptResultSavings"])-StepIndexbeforeFail

            if roundIdx_of_interst >= StartRoundIdx_TraceBack:
                #Failed Round identified
                print("Failed Tracking RollOut")
                failed_file_num = failed_file_num + 1

                roundIdx_of_interst = len(TrackingExp_data["SingleOptResultSavings"])-StepIndexbeforeFail

                SingleOptResult_n_StepsBeforeFail = TrackingExp_data["SingleOptResultSavings"][roundIdx_of_interst]
                print("Get Single Opt Result ", str(StepIndexbeforeFail), "before fail")
                print("Total Number of sucessful Rounds (Start from the Round of starting the alrge slope (idx 2)): ", str(len(TrackingExp_data["SingleOptResultSavings"])-StartRoundIdx_TraceBack))
                print("Index of Round of Interest: ", roundIdx_of_interst)
                roundIdx_of_Interest_List.append(roundIdx_of_interst)

                #Get Init and Terminal COnfig of the plan
                TrackingExp_InitConfig, TrackingExp_TerminalConfig = getInitConfig_in_GlobalFrame_from_SingleOptResult(SingleOptRes=SingleOptResult_n_StepsBeforeFail)

                #Save Unseen State (from local obj tracking exp) and Environment Model
                UnseenState = {}
                UnseenState["TrackingExpPath"] = TrackingExpPath + '/' + TrackingExp_filename
                UnseenState["GroundTruthFile_Path"]=TrackingExp_data["LocalObjSettings"]["GroundTruthTraj"]
                UnseenState["UnseenRoundIdx"]=roundIdx_of_interst
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
                                            "InitRightSurfOrientation":TrackingExp_InitConfig["RightInitSurfOrientation"],
                                            "ContactSurfsVertice":TrackingExp_data["TerrainInfo"]["ContactSurfsVertice"][roundIdx_of_interst:],
                                            "ContactSurfsHalfSpace":TrackingExp_data["TerrainInfo"]["ContactSurfsHalfSpace"][roundIdx_of_interst:],
                                            "ContactSurfsTypes": TrackingExp_data["TerrainInfo"]["ContactSurfsTypes"][roundIdx_of_interst:],
                                            "ContactSurfsNames":TrackingExp_data["TerrainInfo"]["ContactSurfsNames"][roundIdx_of_interst:],
                                            "ContactSurfsTangentX":TrackingExp_data["TerrainInfo"]["ContactSurfsTangentX"][roundIdx_of_interst:],
                                            "ContactSurfsTangentY":TrackingExp_data["TerrainInfo"]["ContactSurfsTangentY"][roundIdx_of_interst:],
                                            "ContactSurfsNorm":TrackingExp_data["TerrainInfo"]["ContactSurfsNorm"][roundIdx_of_interst:],
                                            "ContactSurfsOrientation": TrackingExp_data["TerrainInfo"]["ContactSurfsOrientation"][roundIdx_of_interst:],
                                            "AllPatchesVertices": [TrackingExp_InitConfig["LeftInitSurf"], TrackingExp_InitConfig["RightInitSurf"]] + TrackingExp_data["TerrainInfo"]["ContactSurfsVertice"][roundIdx_of_interst:]
                                            }
                UnseenState["TerrainSettings"] = TrackingExp_data["TerrainSettings"]

                #Save Files
                pickle.dump(UnseenState, open(UnseenStateFolder + '/' + TrackingExp_filename[:-2] + '_' + str(StepIndexbeforeFail) + 'StepBeforeFail' +".p", "wb"))    #Save Data
                
                print(" ")

print("-------Summary--------")
print("Total number of Tracking Exp rollouts being processed: ", total_file_num)
print("Total number of Failed RollOuts: ", failed_file_num)
print(" ")

fig=plt.figure();   ax = fig.gca()
plt.hist(roundIdx_of_Interest_List, bins=5, density = False)
plt.show()