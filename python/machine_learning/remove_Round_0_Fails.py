#NOTE: Need to Copy and Rename RawDataSets

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
#Get working directory from input
cleaningDirectory = sys.argv[1]
print("Clean Failed RollOuts in Folder: \n", cleaningDirectory)

#get all the file names
filenames = os.listdir(cleaningDirectory)

for filename in filenames:
    if ".p" in filename:#a data file
        print("Process: ",filename)

        #Load data
        with open(cleaningDirectory+"/"+filename, 'rb') as f:
            data= pickle.load(f)

        if len(data["SingleOptResultSavings"]) == 0:
            print("Traj length: ",len(data["SingleOptResultSavings"]))
            print("Failed File at Round 0 --- Delete")
            os.remove(cleaningDirectory+"/"+filename)#remove data file
            os.remove(cleaningDirectory+"/"+filename[:-2]+".txt") #remove log file