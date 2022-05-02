# Input 1: Working Folder (.../LargeSlope)
# Input 2: Rollout Folder Name
# Input 3: Destination Folder Name
# INput 4: LargeSLope Angle
# Input 5: Number of files we want to take (count p files only)

# Import Packages
import numpy as np
import os
import pickle
from multicontact_learning_local_objectives.python.machine_learning.ml_utils import *
import matplotlib.pyplot as plt  # Matplotlib
import time
import shutil
import sys

WorkingDirectory = sys.argv[1]
print("Working Directory: \n", WorkingDirectory)

RollOutPath = WorkingDirectory + '/' + sys.argv[2]
print("RollOut Folder Path: ", RollOutPath)

Dest_Path = WorkingDirectory + '/' + sys.argv[3]
print("Destination Folder Path: ", Dest_Path)
# if os.path.isdir(Dest_Path):
#     raise Exception("Destination Folder Exists Already")
# Make the folder
if not (os.path.isdir(Dest_Path)):
    os.mkdir(Dest_Path)

largeslope_angle = sys.argv[4]
angle_indicator = "_"+largeslope_angle
print("LargeSlopeAngle: ", largeslope_angle)


NumFiles_to_Take = int(sys.argv[5])
print("We want to take ", NumFiles_to_Take, "Files")

totalfiles = 0

# get all the file names
filenames = os.listdir(RollOutPath)

file_List = []

for filename in filenames:
    if (".p" in filename) and (angle_indicator in filename):  # a data file
        file_List.append(filename)

# shuffle the list
np.random.shuffle(file_List)

MaxFileNum = np.min((NumFiles_to_Take, len(file_List)))
print("Number of Files we will take: ", MaxFileNum)

for fileIdx in range(MaxFileNum):
    origin_pFileName = file_List[fileIdx]
    origin_txtFileName = file_List[fileIdx][0:-2] + ".txt"
    shutil.move(RollOutPath+"/"+origin_pFileName,
                Dest_Path+"/"+origin_pFileName)
    shutil.move(RollOutPath+"/"+origin_txtFileName,
                Dest_Path+"/"+origin_txtFileName)
