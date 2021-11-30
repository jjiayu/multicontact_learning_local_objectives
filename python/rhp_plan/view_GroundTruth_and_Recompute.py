import numpy as np
from multicontact_learning_local_objectives.python.ocp_build import *
from multicontact_learning_local_objectives.python.terrain_create import *
from multicontact_learning_local_objectives.python.rhp_plan.rhp_utils import *
import multicontact_learning_local_objectives.python.visualization as viz
from multicontact_learning_local_objectives.python.terrain_create import *
from multicontact_learning_local_objectives.python.rhp_plan.get_localobj import *

import sys
import pickle
import time
import copy
import os

import numpy as np
import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D
import multicontact_learning_local_objectives.python.terrain_create.geometry_utils as geometric_utils
import pickle
from multicontact_learning_local_objectives.python.utils import *

fig=plt.figure();   ax = Axes3D(fig)  

recompute_traj = "/home/jiayu/Desktop/MLP_DataSet/LargeSlope_Angle_17_26/ReRollOuts_LargeSlope_20Steps_Start12_Large5_2Steps/Group1_temp1636807427.7535546.p"
TrackingExp_Path = recompute_traj

datatemp = pickle.load((open(recompute_traj,"rb")))

# UnseenStateFilePath = datatemp["TerrainModelPath"]
# idxTemp = UnseenStateFilePath.find("Rubbles_RegretOneStep")
# UnseenStateFilePath = "/media/jiayu/Seagate/"+UnseenStateFilePath[idxTemp:]

# datatemp = pickle.load((open(UnseenStateFilePath,"rb")))
# RollOutTrackingPath = datatemp["TrackingExpPath"]
# idxTemp = RollOutTrackingPath.find("Rubbles_RegretOneStep")

# RollOutTrackingPath = "/media/jiayu/Seagate/" + RollOutTrackingPath[idxTemp:]

# datatemp = pickle.load((open(RollOutTrackingPath,"rb")))

# NLP_traj_Path = datatemp["TerrainModelPath"]

# idxTemp = NLP_traj_Path.find("CleanTrainingSetRollOuts_All_Backup")
# GroundTruthPath = "/media/jiayu/Seagate/"+NLP_traj_Path[idxTemp:]

GroundTruthPath = "/home/jiayu/Desktop/MLP_DataSet/LargeSlope_Angle_17_26/ReRollOuts_LargeSlope_20Steps_Start12_Large5_2Steps/Group1_temp1636807427.7535546.p"

#Result_file_1_path = "/home/jiayu/Desktop/MLP_DataSet/Rubbles/CleanTrainingSetRollOuts_InitialSet/Group1_temp1632242201.9463773.p"
TrackingExp_data = pickle.load(open(TrackingExp_Path,"rb"))

with open(GroundTruthPath, 'rb') as f:
    GroundTruth_data= pickle.load(f)

viz.DisplayResults_Not_Show(TerrainModel = TrackingExp_data["TerrainInfo"], SingleOptResult = None, AllOptResult = TrackingExp_data["SingleOptResultSavings"], fig = fig, ax = ax)
viz.DisplayResults_Not_Show(TerrainModel = GroundTruth_data["TerrainInfo"], SingleOptResult = None, AllOptResult = GroundTruth_data["SingleOptResultSavings"], fig = fig, ax = ax)

plt.show()