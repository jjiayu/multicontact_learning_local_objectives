#Import Packages
import numpy as np
import os
import pickle
from multicontact_learning_local_objectives.python.machine_learning.ml_utils import *
import matplotlib.pyplot as plt #Matplotlib
import time
import shutil
import sys

#--------------------------
#Define Path for Storing Trajectories
#Collect Data Points Path
#workingDirectory = "/home/jiayu/Desktop/multicontact_learning_local_objectives/data/large_slope_flat_patches/"
#workingDirectory = "/afs/inf.ed.ac.uk/group/project/mlp_localobj/Rubbles_and_OneLargeSlope/"
#Get Roll Out path from input
rolloutPath = sys.argv[1]
print("RollOut Path: \n", rolloutPath)

#Define Rollout Path
#rolloutPath = workingDirectory+"RollOuts/"

#get all the file names
filenames = os.listdir(rolloutPath)

#Failing Index Vector
failedIndex = []

#Distance traveled vector
dist_travelled = []

total_file_num = 0
success_file_num = 0

List_StepMade = [] #Define the list containing how many steps makde in the rollouts

for filename in filenames:
    if ".p" in filename:#a data file
        

        total_file_num = total_file_num + 1

        #Load data
        with open(rolloutPath+"/"+filename, 'rb') as f:
            data= pickle.load(f)

        if len(data["SingleOptResultSavings"]) > 0: #add steps made into list of step made
            List_StepMade.append(len(data["SingleOptResultSavings"]))

        if not (len(data["SingleOptResultSavings"]) == data["Num_of_Rounds"]):
            failedIndex.append(len(data["SingleOptResultSavings"])) #The last failed round is not saved, therefore the length of SingleOptResultSavings is the failed index
            
            #Print failed round info
            print("Process: ",filename)
            print("Failed at round: ", len(data["SingleOptResultSavings"])) 
            print("Terrain Model Path: ",data["TerrainModelPath"])
        else:
            success_file_num = success_file_num + 1

        #Get dist traveled 
        if len(data["SingleOptResultSavings"]) > 0: #Failed at 0 steps we dont care
            #Get Init CoM Pos
            init_CoM = data["SingleOptResultSavings"][0]["x_init"]
            #Get Var Index 
            endCoM_var_idx = data["SingleOptResultSavings"][-1]["var_idx"]["Level1_Var_Index"]["x"][-1]
            #Get End CoM
            end_CoM = data["SingleOptResultSavings"][-1]["opt_res"][endCoM_var_idx]
            #Get dist
            dist_temp = end_CoM - init_CoM
            if dist_temp < 0:
                raise Exception("Travelled Negative Distance")
            
            dist_travelled.append(dist_temp)
            
#get statistics
print("Total Number of Roll Outs: ", total_file_num)
print("Successful Roll Outs: ", success_file_num)
for i in range(data["Num_of_Rounds"]):
    print("stop at round " + str(i) + ": " + str(failedIndex.count(i)) + " times")

print(" ")
print("Success Rate Statistics")
successrate = success_file_num/(total_file_num-failedIndex.count(0))*100.0
print("   -Percentage of Successful RollOuts (remove 0 round failures): " + str(np.round(successrate,3)) + "%")

#Compute Confidence Interval for Steps made
confidenceInterval_Mean = np.mean(List_StepMade)
confidenceInterval_Std = np.std(List_StepMade)
upperlower_Limit = 1.96*confidenceInterval_Std/np.sqrt(len(List_StepMade))#Compute with 95% confidence interval
print("   -Confidence Interval:")
print("       -Mean: ", np.round(confidenceInterval_Mean,3))
print("       -Upper and Lower Limit: ", np.round(upperlower_Limit,3))

#Compute Confidence Interval of Distance Travelled
confidenceInterval_Mean = np.mean(dist_travelled)
confidenceInterval_Std = np.std(dist_travelled)
upperlower_Limit = 1.96*confidenceInterval_Std/np.sqrt(len(dist_travelled))#Compute with 95% confidence interval
print("   -Confidence Interval for Distance Travelled:")
print("       -Mean: ", np.round(confidenceInterval_Mean,3))
print("       -Upper and Lower Limit: ", np.round(upperlower_Limit,3))
