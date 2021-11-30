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
#Get Working Directory from input
workingDirectory = sys.argv[1] #i.e. "/home/jiayu/Desktop/multicontact_learning_local_objectives/data/large_slope_flat_patches/"
#Get Roll Out Folder Name from input
RollOutFolderName = sys.argv[2]
print("RollOut Path: \n", workingDirectory + '/' + RollOutFolderName)
#Create TrackingErrorStat Folder
ErroStatFolder = "TrackingErrorStat"



#get all the file names
filenames = os.listdir(rolloutPath)

#Failing Index Vector

total_file_num = 0
success_file_num = 0

for filename in filenames:
    if ".p" in filename:#a data file
        

        total_file_num = total_file_num + 1

        #Load data
        with open(rolloutPath+"/"+filename, 'rb') as f:
            data= pickle.load(f)

        if not (len(data["SingleOptResultSavings"]) == data["Num_of_Rounds"]):
            failedIndex.append(len(data["SingleOptResultSavings"])) #The last failed round is not saved, therefore the length of SingleOptResultSavings is the failed index
            
            #Print failed round info
            print("Process: ",filename)
            print("Failed at round: ", len(data["SingleOptResultSavings"])) 
            print("Terrain Model Path: ",data["TerrainModelPath"])
        else:
            success_file_num = success_file_num + 1
#get statistics
print("Total Number of Roll Outs: ", total_file_num)
print("Successful Roll Outs: ", success_file_num)
for i in range(29):
    print("stop at round " + str(i) + ": " + str(failedIndex.count(i)) + " times")
