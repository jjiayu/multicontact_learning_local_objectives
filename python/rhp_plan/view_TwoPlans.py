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

workingDirectory = "/home/jiayu/Desktop/MLP_DataSet/Rubbles_RegretOneStep"
GroundTruthPath = workingDirectory + "/" + "CleanTrainingSetRollOuts_All_Backup"

#Result_file_1_path = "/home/jiayu/Desktop/MLP_DataSet/Rubbles/CleanTrainingSetRollOuts_InitialSet/Group1_temp1632242201.9463773.p"
TrackingExp_Path = "/home/jiayu/Desktop/MLP_DataSet/Rubbles_RegretOneStep/NN_TrackTraining_All_Aug_1StepBeforeFail_2Time/Group12_temp1633732764.4144151.p"
TrackingExp_data = pickle.load(open(TrackingExp_Path,"rb"))

GroundTruthFileName_StartingIdx = TrackingExp_data["LocalObjSettings"]["GroundTruthTraj"].find("/Group")
GroundTruthFileName = TrackingExp_data["LocalObjSettings"]["GroundTruthTraj"][GroundTruthFileName_StartingIdx+1:]
GroundTruth_file_path = GroundTruthPath + '/' + GroundTruthFileName

with open(GroundTruth_file_path, 'rb') as f:
    GroundTruth_data= pickle.load(f)

viz.DisplayResults_Not_Show(TerrainModel = TrackingExp_data["TerrainInfo"], SingleOptResult = None, AllOptResult = TrackingExp_data["SingleOptResultSavings"], fig = fig, ax = ax)
viz.DisplayResults_Not_Show(TerrainModel = GroundTruth_data["TerrainInfo"], SingleOptResult = None, AllOptResult = GroundTruth_data["SingleOptResultSavings"], fig = fig, ax = ax)

plt.show()