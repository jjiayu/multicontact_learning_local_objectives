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

        if len(data["SingleOptResultSavings"]) >= 2:
            Fail_due_time_Vec = []
            for roundIdx in range(2, len(data["SingleOptResultSavings"])):
                SingleOptRes_Current = data["SingleOptResultSavings"][roundIdx]
                SingleOptRes_Previous = data["SingleOptResultSavings"][roundIdx-1]
                total_proc_time = SingleOptRes_Current["SolverStats"]["t_wall_total"]
                motion_duration_index = SingleOptRes_Previous["var_idx"]["Level1_Var_Index"]["Ts"]
                motion_duration = SingleOptRes_Previous["opt_res"][motion_duration_index[-1]]
                if total_proc_time > motion_duration:
                    Fail_due_time_Vec.append(True)
                elif total_proc_time <= motion_duration:
                    Fail_due_time_Vec.append(False)

            # Decdie the failing index
            if np.sum(Fail_due_time_Vec) == 0:  # Not failing due to time
                fail_due_to_time = False
            else:  # We fail due to time
                fail_due_to_time = True
                Failed_Round_Idx = Fail_due_time_Vec.index(True) + 2

            if fail_due_to_time == True:  # Failed Due to Time
                # do not include the failed round
                List_StepMade.append(Failed_Round_Idx + 1 - 1)
                failedIndex.append(Failed_Round_Idx)
                print("Process: ", filename)
                print("Failed (Due to Large Computation Time) at round: ",
                      Failed_Round_Idx)
                print(data["SingleOptResultSavings"]
                      [Failed_Round_Idx]["SolverStats"]["t_wall_total"])
                print("Terrain Model Path: ", data["TerrainModelPath"])

            elif fail_due_to_time == False:  # Not failing due to time
                # But fail due to convergence
                if (not (len(data["SingleOptResultSavings"]) == data["Num_of_Rounds"])):
                    # one step more than steps made, but index -1
                    failedIndex.append(
                        len(data["SingleOptResultSavings"]) + 1 - 1)
                    List_StepMade.append(len(data["SingleOptResultSavings"]))
                    # Print failed round info
                    print("Process: ", filename)
                    print("Failed at round (Due to Convergence): ",
                          len(data["SingleOptResultSavings"]))
                    print("Terrain Model Path: ", data["TerrainModelPath"])
                else:  # We get successful rollout
                    success_file_num = success_file_num + 1
                    List_StepMade.append(len(data["SingleOptResultSavings"]))

            # Get Computation Time Vector
            for roundIdx in range(2, List_StepMade[-1]):
                SingleOptRes_Current = data["SingleOptResultSavings"][roundIdx]
                SingleOptRes_Previous = data["SingleOptResultSavings"][roundIdx-1]
                total_proc_time = SingleOptRes_Current["SolverStats"]["t_wall_total"]
                motion_duration_index = SingleOptRes_Previous["var_idx"]["Level1_Var_Index"]["Ts"]
                motion_duration = SingleOptRes_Previous["opt_res"][motion_duration_index[-1]]

                computation_time_list.append(total_proc_time)
                motion_duration_list.append(motion_duration)

            # Get dist travelled
            init_CoM = data["SingleOptResultSavings"][0]["x_init"]
            endCoM_var_idx = data["SingleOptResultSavings"][-1]["var_idx"]["Level1_Var_Index"]["x"][-1]
            # Get End CoM
            end_CoM = data["SingleOptResultSavings"][List_StepMade[-1] -
                                                     1]["opt_res"][endCoM_var_idx]
            # Get dist
            dist_temp = end_CoM - init_CoM
            if dist_temp < 0:
                raise Exception("Travelled Negative Distance")

            dist_travelled.append(dist_temp)

# get statistics/media/jiayu/Seagate/RollOut2StepNLP_All
print("Total Number of Roll Outs: ", total_file_num)
print("Successful Roll Outs: ", success_file_num)
for i in range(data["Num_of_Rounds"]):
    print("stop at round " + str(i) + ": " +
          str(failedIndex.count(i)) + " times")

print(" ")
print("Success Rate Statistics")
successrate = success_file_num/(total_file_num-failedIndex.count(0))*100.0
print("   -Percentage of Successful RollOuts (remove 0 round failures): " +
      str(np.round(successrate, 3)) + "%")

# Compute Confidence Interval for Steps made
confidenceInterval_Mean = np.mean(List_StepMade)
confidenceInterval_Std = np.std(List_StepMade)
upperlower_Limit = 1.96*confidenceInterval_Std / \
    np.sqrt(len(List_StepMade))  # Compute with 95% confidence interval
print("   -Confidence Interval:")
print("       -Mean: ", np.round(confidenceInterval_Mean, 3))
print("       -Upper and Lower Limit: ", np.round(upperlower_Limit, 3))

# Compute Confidence Interval of Distance Travelled
confidenceInterval_Mean = np.mean(dist_travelled)
confidenceInterval_Std = np.std(dist_travelled)
upperlower_Limit = 1.96*confidenceInterval_Std / \
    np.sqrt(len(dist_travelled))  # Compute with 95% confidence interval
print("   -Confidence Interval for Distance Travelled:")
print("       -Mean: ", np.round(confidenceInterval_Mean, 3))
print("       -Upper and Lower Limit: ", np.round(upperlower_Limit, 3))

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
plt.hist(computation_time_list, bins=20, density=False)
plt.show()

print(np.max(computation_time_list))
