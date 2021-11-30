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

#Threshold for computation time limit
time_threshold = 0.8

#Define Rollout Path
#rolloutPath = workingDirectory+"RollOuts/"

#get all the file names
filenames = os.listdir(rolloutPath)

total_file_num = 0

#Containers
total_steps_attempted = 0
total_steps_made = 0
total_steps_failed = 0

for filename in filenames:
    if ".p" in filename:#a data file     
        total_file_num = total_file_num + 1

        #Load data
        with open(rolloutPath+"/"+filename, 'rb') as f:
            data= pickle.load(f)

        if len(data["SingleOptResultSavings"]) >= 2: #Overcome the first two steps
            if len(data["SingleOptResultSavings"]) == data["Num_of_Rounds"]: #Completes the whole journey
                total_steps_attempted = total_steps_attempted + data["Num_of_Rounds"] - 2
                total_steps_made = total_steps_made + len(data["SingleOptResultSavings"]) - 2
            elif not (len(data["SingleOptResultSavings"]) == data["Num_of_Rounds"]): #did not complete the whole journey
                total_steps_attempted = total_steps_attempted + len(data["SingleOptResultSavings"]) - 1 + 1 + 1 - 2 #remove the first two steps
                #                                                -----Index of last successful step---  index of failed step; Num of steps attepted till the failed
                total_steps_made = total_steps_made + len(data["SingleOptResultSavings"]) - 2
                total_steps_failed = total_steps_failed + 1
            else:
                raise Exception("Unknown length problem")

print("Stop when reach time Budget (",str(time_threshold),")")
print("Total Number of Steps Attempted to Compute: ", total_steps_attempted)
print("Total Number of Steps Successfully Made: ", str(total_steps_made))
print("    - Percentage: ", str(np.round(total_steps_made/total_steps_attempted*100,3)) + "%")
print("Total Number of Steps Failed to compute within the Time Budget: ", total_steps_failed)
print("    - Percentage: ", str(np.round(total_steps_failed/total_steps_attempted*100,3)) + "%")
