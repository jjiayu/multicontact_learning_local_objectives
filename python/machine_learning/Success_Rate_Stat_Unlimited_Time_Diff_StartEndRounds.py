# $1: Rollout Directory
# $2: Computation time threshold: 1) MotionDuration 2) Numbers (0.8)
# $3: Start Round Num
# $4: End Round Num

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
time_threshold = sys.argv[2]

startRoundNUm = int(sys.argv[3])
endRoundNum = int(sys.argv[4])
#Define Rollout Path
#rolloutPath = workingDirectory+"RollOuts/"

#get all the file names
filenames = os.listdir(rolloutPath)

total_file_num = 0

#Containers
total_steps_attempted = 0
total_steps_made = 0
total_steps_failed = 0
total_steps_failed_dueto_time = 0
total_steps_failed_dueto_convergence = 0

for filename in filenames:
    if ".p" in filename:#a data file     
        total_file_num = total_file_num + 1

        #Load data
        with open(rolloutPath+"/"+filename, 'rb') as f:
            data= pickle.load(f)

        if (len(data["SingleOptResultSavings"]) >= 2): #Overcome the first two steps
            #Check to get stats
            for roundIdx in range(startRoundNUm,np.min([len(data["SingleOptResultSavings"]),endRoundNum+1])):
                #Get data for current round
                total_steps_attempted = total_steps_attempted + 1
                SingleOptRes_Current = data["SingleOptResultSavings"][roundIdx]
                total_proc_time = SingleOptRes_Current["SolverStats"]["t_wall_total"] #t_proc_total
                #Get motion duration from previous round
                SingleOptRes_Previous = data["SingleOptResultSavings"][roundIdx-1]
                motion_duration_index = SingleOptRes_Previous["var_idx"]["Level1_Var_Index"]["Ts"]
                motion_duration = SingleOptRes_Previous["opt_res"][motion_duration_index[-1]]

                if time_threshold == "MotionDuration":
                    time_limit = motion_duration
                else:
                    time_limit = float(time_threshold)

                if total_proc_time <= time_limit: #Smaller than threshold
                    total_steps_made = total_steps_made + 1
                else: #Larger than threshold
                    total_steps_failed = total_steps_failed + 1
                    total_steps_failed_dueto_time = total_steps_failed_dueto_time + 1
            
        if (len(data["SingleOptResultSavings"]) > startRoundNUm+1) and (len(data["SingleOptResultSavings"]) < endRoundNum+1): #The last step failed due to some convergence reason
            total_steps_attempted = total_steps_attempted + 1
            total_steps_failed = total_steps_failed + 1
            total_steps_failed_dueto_convergence = total_steps_failed_dueto_convergence + 1

print("Keeps Rolling out even exceeds time limit")
print("Total Number of Steps Attempted to Compute: ", total_steps_attempted)
print("Total Number of Steps Computed within time budge (", str(time_threshold), "): ", str(total_steps_made))
print("    - Percentage: ", str(np.round(total_steps_made/total_steps_attempted*100,3)) + "%")
print("Total Number of Steps Failed to compute (Either within the time budget or convergence problem): ", total_steps_failed)
print("    - Percentage: ", str(np.round(total_steps_failed/total_steps_attempted*100,3)) + "%")
print("(Break Down) Total Number of Steps failed to compute due to Running out of Time: ", total_steps_failed_dueto_time)
print("    - Percentage: ", str(np.round(total_steps_failed_dueto_time/total_steps_attempted*100,3)) + "%")
print("(Break Down) Total Number of Steps failed to compute due to Convergence: ", total_steps_failed_dueto_convergence)
print("    - Percentage: ", str(np.round(total_steps_failed_dueto_convergence/total_steps_attempted*100,3)) + "%")
