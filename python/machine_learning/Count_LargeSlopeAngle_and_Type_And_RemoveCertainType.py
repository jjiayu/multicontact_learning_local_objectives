#Input 1: RollOut Folder Name
#Input 2: Idx Number of tge large slope
#Input 3: Remove Type: X_negative, X_positive

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

large_slope_idx = int(sys.argv[2])
print("Large Slope at: ", large_slope_idx)

removeType = sys.argv[3]

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

angle_list = []
angle_list_x_pos = []
angle_list_x_neg = []
angle_list_y_pos = []
angle_list_y_neg = []

for filename in filenames:
    if ".p" in filename:#a data file
        

        total_file_num = total_file_num + 1

        #Load data
        with open(rolloutPath+"/"+filename, 'rb') as f:
            data= pickle.load(f)

        terrain_model = data["TerrainModel"]

        rotationtype = getSurfaceType(terrain_model[large_slope_idx+2])
        rotationangle=np.absolute(getTerrainRotationAngle(terrain_model[large_slope_idx+2]))

        angle_list.append(rotationangle)

        if rotationtype =="X_negative":
            angle_list_x_neg.append(rotationangle)
        elif rotationtype =="X_positive":
            angle_list_x_pos.append(rotationangle)
        elif rotationtype =="Y_negative":
            angle_list_y_neg.append(rotationangle)
        elif rotationtype =="Y_positive":
            angle_list_y_pos.append(rotationangle)

        if rotationtype == removeType:
            print("Failed File --- Delete, bnecasue type, ", rotationtype, "and we want to remove ", removeType)
            os.remove(rolloutPath+"/"+filename)#remove data file
            os.remove(rolloutPath+"/"+filename[:-2]+".txt") #remove log file

fig=plt.figure();   ax = fig.gca()
plt.hist(angle_list, bins=10, density = False)
ax.set_xlim([17,26])
plt.title("All")
plt.show()

fig=plt.figure();   ax = fig.gca()
plt.hist(angle_list_x_neg, bins=10, density = False)
ax.set_xlim([17,26])
plt.show()


fig=plt.figure();   ax = fig.gca()
plt.hist(angle_list_x_pos, bins=10, density = False)
ax.set_xlim([17,26])
plt.show()

fig=plt.figure();   ax = fig.gca()
plt.hist(angle_list_y_neg, bins=10, density = False)
ax.set_xlim([17,26])
plt.show()


fig=plt.figure();   ax = fig.gca()
plt.hist(angle_list_y_pos, bins=10, density = False)
ax.set_xlim([17,26])
plt.show()

print("Rotation Around X Positive: ", len(angle_list_x_pos))
print("Rotation Around X Negative: ", len(angle_list_x_neg))
print("Rotation Around Y Positive: ", len(angle_list_y_pos))
print("Rotation Around Y Negative: ", len(angle_list_y_neg))