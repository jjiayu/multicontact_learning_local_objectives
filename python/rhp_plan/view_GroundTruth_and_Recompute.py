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

#recompute_traj = "/home/jiayu/Desktop/MLP_DataSet/Testing_Largeslope_TimeTrack_Angle_17_26/NN_TrackTestAll_LargeSlope/Group1_temp1642014142.3924603.p"
recompute_traj = "/home/jiayu/Desktop/MLP_DataSet/GroundTruthTraj/uneven_plan_changing_env.p"#"/home/jiayu/Desktop/MLP_DataSet/GroundTruthTraj/flat_temp_short_duration.p"
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

#GroundTruthPath = "/home/jiayu/Desktop/MLP_DataSet/Testing_Largeslope_TimeTrack_Angle_17_26/CleanValidationRollOut/Group1_temp1639666706.7087774_Y_positive_angle_24.7.p"
#GroundTruthPath = "/home/jiayu/Desktop/MLP_DataSet/rhp_motion_plans/rhp_plan.p"#"/home/jiayu/Desktop/MLP_DataSet/GroundTruthTraj/flat_temp_long_phase.p"

GroundTruthPath = "/home/jiayu/Desktop/MLP_DataSet/GroundTruthTraj/uneven_plan_unchanged.p"

#Result_file_1_path = "/home/jiayu/Desktop/MLP_DataSet/Rubbles/CleanTrainingSetRollOuts_InitialSet/Group1_temp1632242201.9463773.p"
TrackingExp_data = pickle.load(open(TrackingExp_Path,"rb"))

with open(GroundTruthPath, 'rb') as f:
    GroundTruth_data= pickle.load(f)

viz.DisplayResults_Not_Show(TerrainModel = TrackingExp_data["TerrainInfo"], SingleOptResult = None, AllOptResult = TrackingExp_data["SingleOptResultSavings"], fig = fig, ax = ax)
viz.DisplayResults_Not_Show(TerrainModel = GroundTruth_data["TerrainInfo"], SingleOptResult = None, AllOptResult = GroundTruth_data["SingleOptResultSavings"], fig = fig, ax = ax)

plt.show()