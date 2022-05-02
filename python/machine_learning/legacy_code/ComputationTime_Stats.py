# Import Packages
import numpy as np
import os
import pickle
from multicontact_learning_local_objectives.python.machine_learning.ml_utils import *
import matplotlib.pyplot as plt  # Matplotlib
import time
import shutil
import sys

# --------------------------
# Define Path for Storing Trajectories
# Collect Data Points Path
#workingDirectory = "/home/jiayu/Desktop/multicontact_learning_local_objectives/data/large_slope_flat_patches/"
#workingDirectory = "/afs/inf.ed.ac.uk/group/project/mlp_localobj/Rubbles_and_OneLargeSlope/"
# Get Roll Out path from input
rolloutPath = sys.argv[1]
print("RollOut Path: \n", rolloutPath)

# Define Rollout Path
#rolloutPath = workingDirectory+"RollOuts/"

# get all the file names
filenames = os.listdir(rolloutPath)

# Failing Index Vector
failedIndex = []

# Distance traveled vector
dist_travelled = []

total_file_num = 0
success_file_num = 0

List_StepMade = []  # Define the list containing how many steps makde in the rollouts

computation_time_list = []
motion_duration_list = []

for filename in filenames:
    if ".p" in filename:  # a data file
        total_file_num = total_file_num + 1

        # Load data
        with open(rolloutPath+"/"+filename, 'rb') as f:
            data = pickle.load(f)

        if len(data["SingleOptResultSavings"]) == data["Num_of_Rounds"]:
            for roundIdx in range(2, len(data["SingleOptResultSavings"])):
                SingleOptRes_Current = data["SingleOptResultSavings"][roundIdx]
                SingleOptRes_Previous = data["SingleOptResultSavings"][roundIdx-1]
                total_proc_time = SingleOptRes_Current["SolverStats"]["t_proc_total"]
                motion_duration_index = SingleOptRes_Previous["var_idx"]["Level1_Var_Index"]["Ts"]
                motion_duration = SingleOptRes_Previous["opt_res"][motion_duration_index[-1]]

                computation_time_list.append(total_proc_time)
                motion_duration_list.append(motion_duration)

# Compute Mean and Std
computation_time_mean = np.mean(computation_time_list)
computation_time_std = np.std(computation_time_list)

motion_duratione_mean = np.mean(motion_duration_list)
motion_duration_std = np.std(motion_duration_list)

print("Computation Time Mean: ", computation_time_mean)
print("Computation Time Std: ", computation_time_std)

print("Motion Duration Mean: ", motion_duratione_mean)
print("Motion Duration Std: ", motion_duration_std)

computation_time_upperlower_limit = 1.96 * \
    computation_time_std/np.sqrt(len(computation_time_list))

print("   -Confidence Interval on Computation time:")
print("       -Mean: ", np.round(computation_time_mean, 3))
print("       -Upper and Lower Limit: ",
      np.round(computation_time_upperlower_limit, 3))

# Plot Histogram
plt.hist(computation_time_list, bins=10, density=False)
plt.show()
