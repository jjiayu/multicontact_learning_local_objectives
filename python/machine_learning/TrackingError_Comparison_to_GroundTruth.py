
# Import packages
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

np.set_printoptions(precision=4)

# --------------
# Set up for Directories
#   Define Working Directory
workingDirectory = "/home/jiayu/Desktop/MLP_DataSet/Rubbles"
print("Working folder: ", workingDirectory)
#   Define Tracking RollOuts Path
TrackingExpPath = workingDirectory + "/" + "NN_TrackTraining_InitialSet"
if not (os.path.isdir(TrackingExpPath)):
    raise Exception("Fault Tracking Exp Path")
else:
    print("Tracking Exp RollOut folder: ", TrackingExpPath)
#   Define Ground Truth RollOuts Path
GroundTruthPath = workingDirectory + "/" + "CleanTrainingSetRollOuts_InitialSet"
if not (os.path.isdir(GroundTruthPath)):
    raise Exception("Fault Ground Truth RollOut Path")
else:
    print("Ground Truth RollOut folder: ", GroundTruthPath)

#   Define Number of file proccessed
total_file_num = 0
failed_file_num = 0
success_file_num = 0

#   Get File names in the Local Obj Tracking Exp Folder
TrackingExpfilenames = os.listdir(TrackingExpPath)

success_error_all = []
failed_error_all = []

# Loop over all the tracking exp files
for TrackingExp_filename in TrackingExpfilenames:

    if ".p" in TrackingExp_filename:  # a data file

        print("Process Tracking Exp File: ", TrackingExp_filename)
        total_file_num = total_file_num + 1  # Process one file

        # Load Tracking Exp data
        with open(TrackingExpPath + '/' + TrackingExp_filename, 'rb') as f:
            TrackingExp_data = pickle.load(f)

        # Load Ground Truth data
        #   Get Ground Truth File name
        GroundTruthFileName_StartingIdx = TrackingExp_data["LocalObjSettings"]["GroundTruthTraj"].find(
            "/Group")
        GroundTruthFileName = TrackingExp_data["LocalObjSettings"][
            "GroundTruthTraj"][GroundTruthFileName_StartingIdx+1:]
        GroundTruth_file_path = GroundTruthPath + '/' + GroundTruthFileName
        with open(GroundTruth_file_path, 'rb') as f:
            GroundTruth_data = pickle.load(f)

        # Compare Error in the Global Frame (Environment is the same, only compare state)
        # also need to filter out the cases with 0 step failed
        if len(TrackingExp_data["SingleOptResultSavings"]) < TrackingExp_data["Num_of_Rounds"] and len(TrackingExp_data["SingleOptResultSavings"]) > 0:
            # Failed Tracking Exp
            print("- Failed Tracking Exp")
            failed_file_num = failed_file_num + 1

            # Loop Over all the rounds
            err_list_temp = []
            for roundIdx in range(len(TrackingExp_data["SingleOptResultSavings"])):
                TrackingExp_SingleOptResult = TrackingExp_data["SingleOptResultSavings"][roundIdx]
                TrackingExp_InitConfig_Temp, TrackingExp_TerminalConfig_Temp = getInitConfig_in_GlobalFrame_from_SingleOptResult(
                    SingleOptRes=TrackingExp_SingleOptResult)
                TrackingExp_State = np.array([TrackingExp_InitConfig_Temp["x_init"], TrackingExp_InitConfig_Temp["y_init"], TrackingExp_InitConfig_Temp["z_init"],
                                              TrackingExp_InitConfig_Temp["xdot_init"], TrackingExp_InitConfig_Temp[
                                                  "ydot_init"], TrackingExp_InitConfig_Temp["zdot_init"],
                                              TrackingExp_InitConfig_Temp["Lx_init"], TrackingExp_InitConfig_Temp[
                                                  "Ly_init"], TrackingExp_InitConfig_Temp["Lz_init"],
                                              TrackingExp_InitConfig_Temp["PLx_init"], TrackingExp_InitConfig_Temp[
                                                  "PLy_init"], TrackingExp_InitConfig_Temp["PLz_init"],
                                              TrackingExp_InitConfig_Temp["PRx_init"], TrackingExp_InitConfig_Temp["PRy_init"], TrackingExp_InitConfig_Temp["PRz_init"]])

                GroundTruth_SingleOptResult = GroundTruth_data["SingleOptResultSavings"][roundIdx]
                GroundTruth_InitConfig_Temp, GroundTruth_TerminalConfig_Temp = getInitConfig_in_GlobalFrame_from_SingleOptResult(
                    SingleOptRes=GroundTruth_SingleOptResult)
                GroundTruth_State = np.array([GroundTruth_InitConfig_Temp["x_init"], GroundTruth_InitConfig_Temp["y_init"], GroundTruth_InitConfig_Temp["z_init"],
                                              GroundTruth_InitConfig_Temp["xdot_init"], GroundTruth_InitConfig_Temp[
                                                  "ydot_init"], GroundTruth_InitConfig_Temp["zdot_init"],
                                              GroundTruth_InitConfig_Temp["Lx_init"], GroundTruth_InitConfig_Temp[
                                                  "Ly_init"], GroundTruth_InitConfig_Temp["Lz_init"],
                                              GroundTruth_InitConfig_Temp["PLx_init"], GroundTruth_InitConfig_Temp[
                                                  "PLy_init"], GroundTruth_InitConfig_Temp["PLz_init"],
                                              GroundTruth_InitConfig_Temp["PRx_init"], GroundTruth_InitConfig_Temp["PRy_init"], GroundTruth_InitConfig_Temp["PRz_init"]])

                # Compute Error
                error_vec = np.linalg.norm(
                    TrackingExp_State - GroundTruth_State)
                err_list_temp.append(error_vec)

            failed_error_all.append(err_list_temp)
            # print(err_list_temp)
            # fig=plt.figure();   ax = fig.gca()
            # plt.plot(err_list_temp)
            # plt.show()
            #             TrackingExp_InitConfig, TrackingExp_TerminalConfig = getInitConfig_in_GlobalFrame_from_SingleOptResult(SingleOptRes=SingleOptResult_n_StepsBeforeFail)

        elif len(TrackingExp_data["SingleOptResultSavings"]) == TrackingExp_data["Num_of_Rounds"]:
            # Successful Tracking Exp
            print("- Successful Tracking Exp")
            success_file_num = success_file_num + 1

            # Loop Over all the rounds
            err_list_temp = []
            for roundIdx in range(len(TrackingExp_data["SingleOptResultSavings"])):
                TrackingExp_SingleOptResult = TrackingExp_data["SingleOptResultSavings"][roundIdx]
                TrackingExp_InitConfig_Temp, TrackingExp_TerminalConfig_Temp = getInitConfig_in_GlobalFrame_from_SingleOptResult(
                    SingleOptRes=TrackingExp_SingleOptResult)
                TrackingExp_State = np.array([TrackingExp_InitConfig_Temp["x_init"], TrackingExp_InitConfig_Temp["y_init"], TrackingExp_InitConfig_Temp["z_init"],
                                              TrackingExp_InitConfig_Temp["xdot_init"], TrackingExp_InitConfig_Temp[
                                                  "ydot_init"], TrackingExp_InitConfig_Temp["zdot_init"],
                                              TrackingExp_InitConfig_Temp["Lx_init"], TrackingExp_InitConfig_Temp[
                                                  "Ly_init"], TrackingExp_InitConfig_Temp["Lz_init"],
                                              TrackingExp_InitConfig_Temp["PLx_init"], TrackingExp_InitConfig_Temp[
                                                  "PLy_init"], TrackingExp_InitConfig_Temp["PLz_init"],
                                              TrackingExp_InitConfig_Temp["PRx_init"], TrackingExp_InitConfig_Temp["PRy_init"], TrackingExp_InitConfig_Temp["PRz_init"]])

                GroundTruth_SingleOptResult = GroundTruth_data["SingleOptResultSavings"][roundIdx]
                GroundTruth_InitConfig_Temp, GroundTruth_TerminalConfig_Temp = getInitConfig_in_GlobalFrame_from_SingleOptResult(
                    SingleOptRes=GroundTruth_SingleOptResult)
                GroundTruth_State = np.array([GroundTruth_InitConfig_Temp["x_init"], GroundTruth_InitConfig_Temp["y_init"], GroundTruth_InitConfig_Temp["z_init"],
                                              GroundTruth_InitConfig_Temp["xdot_init"], GroundTruth_InitConfig_Temp[
                                                  "ydot_init"], GroundTruth_InitConfig_Temp["zdot_init"],
                                              GroundTruth_InitConfig_Temp["Lx_init"], GroundTruth_InitConfig_Temp[
                                                  "Ly_init"], GroundTruth_InitConfig_Temp["Lz_init"],
                                              GroundTruth_InitConfig_Temp["PLx_init"], GroundTruth_InitConfig_Temp[
                                                  "PLy_init"], GroundTruth_InitConfig_Temp["PLz_init"],
                                              GroundTruth_InitConfig_Temp["PRx_init"], GroundTruth_InitConfig_Temp["PRy_init"], GroundTruth_InitConfig_Temp["PRz_init"]])

                # Compute Error
                error_vec = np.linalg.norm(
                    TrackingExp_State - GroundTruth_State)
                err_list_temp.append(error_vec)
            print(err_list_temp)

            success_error_all.append(err_list_temp)

# Compute Mean error for success
success_mean_error = np.mean(success_error_all, axis=0)
success_std_error = np.std(success_error_all, axis=0)
stepIdx = np.array([i for i in range(30)])
print("mean error for sucess: ", success_mean_error)
print("std error for sucess: ", success_std_error)
fig = plt.figure()
ax = fig.gca()
plt.errorbar(stepIdx, success_mean_error, success_std_error)
plt.show()


# Compute Mean error for failed
failed_mean_error = []
failed_std_error = []

for i in range(30):
    templist = []
    for j in range(len(failed_error_all)):
        if i <= len(failed_error_all[j]) - 1:
            templist.append(failed_error_all[j][i])

    temp_mean = np.mean(templist)
    temp_std = np.std(templist)

    failed_mean_error.append(temp_mean)
    failed_std_error.append(temp_std)

failed_mean_error = np.array(failed_mean_error)
failed_std_error = np.array(failed_std_error)

print("mean error for fail: ", failed_mean_error)
print("std error for fail: ", failed_std_error)
fig = plt.figure()
ax = fig.gca()
plt.errorbar(stepIdx, failed_mean_error, failed_std_error)
plt.show()


#         #Check if the current round is a failued round or success round
#         if len(TrackingExp_data["SingleOptResultSavings"]) < TrackingExp_data["Num_of_Rounds"] and len(TrackingExp_data["SingleOptResultSavings"]) > 0: #also need to filter out the cases with 0 step failed
#             #Failed Round identified
#             print("Failed Tracking RollOut")
#             failed_file_num = failed_file_num + 1

#             roundIdx_of_interst = len(TrackingExp_data["SingleOptResultSavings"])-StepIndexbeforeFail

#             SingleOptResult_n_StepsBeforeFail = TrackingExp_data["SingleOptResultSavings"][roundIdx_of_interst]
#             print("Get Single Opt Result ", str(StepIndexbeforeFail), "before fail")
#             print("Total Number of sucessful Rounds: ", str(len(TrackingExp_data["SingleOptResultSavings"])))
#             print("Index of Round of Interest: ", roundIdx_of_interst)

#             #Get Init and Terminal COnfig of the plan
#             TrackingExp_InitConfig, TrackingExp_TerminalConfig = getInitConfig_in_GlobalFrame_from_SingleOptResult(SingleOptRes=SingleOptResult_n_StepsBeforeFail)

#             #Save Unseen State (from local obj tracking exp) and Environment Model
#             UnseenState = {}
#             UnseenState["TrackingExpPath"] = TrackingExpPath + '/' + TrackingExp_filename
#             UnseenState["InitConfig"] = TrackingExp_InitConfig
#             UnseenState["TerrainInfo"] = {"InitLeftSurfVertice":     TrackingExp_InitConfig["LeftInitSurf"],
#                                           "InitLeftSurfTangentX":    TrackingExp_InitConfig["PL_init_TangentX"],
#                                           "InitLeftSurfTangentY":    TrackingExp_InitConfig["PL_init_TangentY"],
#                                           "InitLeftSurfNorm":        TrackingExp_InitConfig["PL_init_Norm"],
#                                           "InitLeftSurfOrientation": TrackingExp_InitConfig["LeftInitSurfOrientation"],
#                                           "InitRightSurfVertice":    TrackingExp_InitConfig["RightInitSurf"],
#                                           "InitRightSurfTangentX":   TrackingExp_InitConfig["PR_init_TangentX"],
#                                           "InitRightSurfTangentY":   TrackingExp_InitConfig["PR_init_TangentY"],
#                                           "InitRightSurfNorm":       TrackingExp_InitConfig["PR_init_Norm"],
#                                           "InitRightSurfOrientation":TrackingExp_InitConfig["RightInitSurfOrientation"],
#                                           "ContactSurfsVertice":TrackingExp_data["TerrainInfo"]["ContactSurfsVertice"][roundIdx_of_interst:],
#                                           "ContactSurfsHalfSpace":TrackingExp_data["TerrainInfo"]["ContactSurfsHalfSpace"][roundIdx_of_interst:],
#                                           "ContactSurfsTypes": TrackingExp_data["TerrainInfo"]["ContactSurfsTypes"][roundIdx_of_interst:],
#                                           "ContactSurfsNames":TrackingExp_data["TerrainInfo"]["ContactSurfsNames"][roundIdx_of_interst:],
#                                           "ContactSurfsTangentX":TrackingExp_data["TerrainInfo"]["ContactSurfsTangentX"][roundIdx_of_interst:],
#                                           "ContactSurfsTangentY":TrackingExp_data["TerrainInfo"]["ContactSurfsTangentY"][roundIdx_of_interst:],
#                                           "ContactSurfsNorm":TrackingExp_data["TerrainInfo"]["ContactSurfsNorm"][roundIdx_of_interst:],
#                                           "ContactSurfsOrientation": TrackingExp_data["TerrainInfo"]["ContactSurfsOrientation"][roundIdx_of_interst:],
#                                           "AllPatchesVertices": [TrackingExp_InitConfig["LeftInitSurf"], TrackingExp_InitConfig["RightInitSurf"]] + TrackingExp_data["TerrainInfo"]["ContactSurfsVertice"][roundIdx_of_interst:]
#                                         }
#             UnseenState["TerrainSettings"] = TrackingExp_data["TerrainSettings"]

#             #Save Files
#             pickle.dump(UnseenState, open(UnseenStateFolder + '/' + TrackingExp_filename[:-2] + '_' + str(StepIndexbeforeFail) + 'StepBeforeFail' +".p", "wb"))    #Save Data

#             print(" ")

# print("-------Summary--------")
# print("Total number of Tracking Exp rollouts being processed: ", total_file_num)
# print("Total number of Failed RollOuts: ", failed_file_num)
# print(" ")
