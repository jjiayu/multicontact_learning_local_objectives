#Input 1: RollOut Folder Name
#Input 2: Start Idx of the Round
#Input 3: End Idx of the Round

#Import Packages
import numpy as np
import os
import pickle
from multicontact_learning_local_objectives.python.machine_learning.ml_utils import *
from multicontact_learning_local_objectives.python.terrain_create import *
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

startingIdx = int(sys.argv[2])
print("Get RollOut from Round: ", startingIdx)
EndIdx = int(sys.argv[3])
print("End Getting RollOut from Round: ", EndIdx)

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

InitialDoubleSupport_List = []
SingleSupport_List = []
DoubleSupport_List = []

for filename in filenames:
    if ".p" in filename:#a data file
        
        #Load data
        with open(rolloutPath+"/"+filename, 'rb') as f:
            data= pickle.load(f)

        if len(data["SingleOptResultSavings"]) > 2:
            for roundNum in range(startingIdx,np.min([len(data["SingleOptResultSavings"]),EndIdx+1])):#range(data["Num_of_Rounds"]):
                #print("   Process Round: ", roundNum)
                #Get Single Optimization Result of current result
                singleOptResult = data["SingleOptResultSavings"][roundNum]

                #   Get fixed variables for the first level
                decision_var_index_level1 = singleOptResult["var_idx"]["Level1_Var_Index"]

                timing_lv1_fixed = singleOptResult["opt_res"][decision_var_index_level1["Ts"][0]:decision_var_index_level1["Ts"][1]+1]

                InitialDoubleSupport_List.append(timing_lv1_fixed[0]) 
                SingleSupport_List.append(timing_lv1_fixed[1]) 
                DoubleSupport_List.append(timing_lv1_fixed[2]) 

                #   Get fixed variables for the first level
                if data["NumLookAhead"] > 1:

                    decision_var_index_level2 = singleOptResult["var_idx"]["Level2_Var_Index"]
                    x_opt_level2 = singleOptResult["opt_res"][decision_var_index_level1["Ts"][1]+1:]
                    timing_lv2_fixed = x_opt_level2[decision_var_index_level2["Ts"][0]:decision_var_index_level2["Ts"][1]+1]

                    for timingidx in range(len(timing_lv2_fixed)/3):
                        InitialDoubleSupport_List.append(timing_lv1_fixed[3*timingidx+0]) 
                        SingleSupport_List.append(timing_lv1_fixed[3*timingidx+1]) 
                        DoubleSupport_List.append(timing_lv1_fixed[3*timingidx+2]) 
            
fig=plt.figure();   ax = fig.gca()
plt.hist(InitialDoubleSupport_List, bins=100, density = False)
plt.show()

fig=plt.figure();   ax = fig.gca()
plt.hist(SingleSupport_List, bins=100, density = False)
plt.show()


fig=plt.figure();   ax = fig.gca()
plt.hist(DoubleSupport_List, bins=100, density = False)
plt.show()

