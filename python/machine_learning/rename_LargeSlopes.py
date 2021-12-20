#Input 1: Working Folder (.../LargeSlope)
#Input 2: Rollout Folder Name
#Input 3: Destination Folder Name
#INput 4: Type of the large slope


#Import Packages
import numpy as np
import os
import pickle
from multicontact_learning_local_objectives.python.machine_learning.ml_utils import *
import matplotlib.pyplot as plt #Matplotlib
import time
import shutil
import sys

WorkingDirectory = sys.argv[1]
print("Working Directory: \n", WorkingDirectory)

RollOutPath = WorkingDirectory + '/' + sys.argv[2]
print("RollOut Folder Path: ", RollOutPath)

Dest_Path = WorkingDirectory + '/' + sys.argv[3]
print("Destination Folder Path: ", Dest_Path)
if os.path.isdir(Dest_Path):
    raise Exception("Destination Folder Exists Already")
#Make the folder
if not (os.path.isdir(Dest_Path)):
    os.mkdir(Dest_Path)

largeslope_type = sys.argv[4]
print("Type of the large slope: ", largeslope_type)

totalfiles = 0

#get all the file names
filenames = os.listdir(RollOutPath)

for filename in filenames:
    if ".p" in filename:#a data file

        #Load data
        with open(RollOutPath+"/"+filename, 'rb') as f:
            data= pickle.load(f)

        totalfiles = totalfiles + 1
        #Get terrain path
        terrainPath = data["TerrainModelPath"]
        #Get the appendix of the file name
        appendedNameIdx = terrainPath.find(largeslope_type+"_angle")
        appendedName = terrainPath[appendedNameIdx:-2]
        
        #Get File Names
        origin_pFileName = filename
        origin_txtFileName = filename[0:-2] + ".txt"
        dest_pFileName = filename[0:-2]+"_"+appendedName+".p"
        dest_txtFileName = filename[0:-2]+"_"+appendedName+".txt"
        print("------------------------------------------")
        print("Original pFile Name: ", origin_pFileName)
        print("Original txtFile Name: ", origin_txtFileName)
        print("Destination pFile Name: ", dest_pFileName)
        print("Destination txtFile Name: ", dest_txtFileName)
        print("------------------------------------------")

        #Copy
        #For pFile
        pFile_src_path = RollOutPath + "/" + origin_pFileName
        pFile_dest_path = Dest_Path + "/" + dest_pFileName
        shutil.copy(pFile_src_path, pFile_dest_path)

        #For txtFile
        txtFile_src_path = RollOutPath + "/" + origin_txtFileName
        txtFile_dest_path = Dest_Path + "/" + dest_txtFileName
        shutil.copy(txtFile_src_path, txtFile_dest_path)

print("Total Number of File Processed: ", totalfiles)


        

        