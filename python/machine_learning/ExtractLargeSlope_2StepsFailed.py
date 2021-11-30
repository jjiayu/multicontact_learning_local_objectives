#Input 1: Working Directory Name (Full Path)
#Input 2: RollOut of 2Steps lookahead
#Input 3: RollOut of NSteps Lookahead
#Input 4: Folder Contains The Extracted Env 

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

workingDirectory = sys.argv[1]
print("Working Directory: \n", workingDirectory)

LookAhead_2Steps_Folder = sys.argv[2]
LookAhead_2Steps_Path = workingDirectory + '/' + LookAhead_2Steps_Folder
print("2Step LookAhead Directory: \n", LookAhead_2Steps_Path)

LookAhead_NSteps_Folder = sys.argv[3]
LookAhead_NSteps_Path = workingDirectory + '/' + LookAhead_NSteps_Folder
print("NStep LookAhead Directory: \n", LookAhead_NSteps_Path)

ExtractedEnv_Folder = sys.argv[4]
ExtractedEnv_Path = workingDirectory + '/' + ExtractedEnv_Folder

if os.path.isdir(ExtractedEnv_Path): #NOTE: Report if we already has a folder with the same name, maybe we need to backup
    raise Exception("Folder with the same name already exists, Backup the original folder and re-name the folder we want to use")
else:
    os.mkdir(ExtractedEnv_Path)
    print("Folder to Store Extracted Env: ", ExtractedEnv_Path)

#get all the file names
filenames = os.listdir(LookAhead_2Steps_Path)

for filename in filenames:
    if ".p" in filename:#a data file
        
        #Load data
        with open(LookAhead_2Steps_Path+"/"+filename, 'rb') as f:
            data= pickle.load(f)

        if len(data["SingleOptResultSavings"]) > 2 and not (len(data["SingleOptResultSavings"]) == data["Num_of_Rounds"]): #made the first 2 (quasi-flat) steps but failed eventually
            Env_File_Path = data["TerrainModelPath"]
            Env_File_Name_Idx = Env_File_Path.find("Group")
            #Cope p file
            src_path = LookAhead_NSteps_Path + '/' + Env_File_Path[Env_File_Name_Idx:]
            dst_path = ExtractedEnv_Path
            print("Copy: ", src_path)
            shutil.copy(src_path, dst_path)
            #Cope txt File
            src_path = LookAhead_NSteps_Path + '/' + Env_File_Path[Env_File_Name_Idx:][:-2] + ".txt"
            dst_path = ExtractedEnv_Path
            print("Copy: ", src_path)
            shutil.copy(src_path, dst_path)
            
