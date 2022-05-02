from sklearn.neighbors import NearestNeighbors
import numpy as np
from multicontact_learning_local_objectives.python.ocp_build import *
from multicontact_learning_local_objectives.python.terrain_create import *
from multicontact_learning_local_objectives.python.rhp_plan.rhp_utils import *
import multicontact_learning_local_objectives.python.visualization as viz
from multicontact_learning_local_objectives.python.terrain_create import *
from multicontact_learning_local_objectives.python.rhp_plan.get_localobj import *
from multicontact_learning_local_objectives.python.machine_learning.ml_utils import *

import sys
import pickle
import time
import copy
import os

import numpy as np
import matplotlib.pyplot as plt  # Matplotlib
from mpl_toolkits.mplot3d import Axes3D
import multicontact_learning_local_objectives.python.terrain_create.geometry_utils as geometric_utils
import pickle
from multicontact_learning_local_objectives.python.utils import *

WorkingDirectory = "/media/jiayu/Seagate/Rubbles_AddVarSteps_1StepbeforeFail_RemovebyClip"

# Get files
Recompute_Path = "/media/jiayu/Seagate/Rubbles_AddVarSteps_1StepbeforeFail_RemovebyClip/RollOuts_20Steps_from_1StepbeforeFail_TrackTrainingAll_InitialSet/Group2_temp1635347902.69429.p"
Recompute_data = pickle.load(open(Recompute_Path, "rb"))

UnseenStatePath = Recompute_data["TerrainModelPath"]
idxTemp = UnseenStatePath.find("Unseen")
UnseenStatePath = WorkingDirectory + "/" + UnseenStatePath[idxTemp:]
UnseenState_data = pickle.load(open(UnseenStatePath, "rb"))

GroundTruthFilePath = UnseenState_data["GroundTruthFile_Path"]
idxTemp = GroundTruthFilePath.find("Clean")
GroundTruthFilePath = WorkingDirectory + "/" + GroundTruthFilePath[idxTemp:]
GroundTruth_data = pickle.load(open(GroundTruthFilePath, "rb"))

print(GroundTruthFilePath)

# Load DataSet
workingDirectory = "/media/jiayu/Seagate/Rubbles_AddVarSteps_1StepbeforeFail_RemovebyClip"
# NOTE: need to have "/" at the end
print("Double Check we provide the Correct Traj Path: \n", workingDirectory)
# Define dataset folder
#DataSetPath = workingDirectory + "/DataSet_Large_Standarized"
DataSetPath = workingDirectory + "/DataSet/"
TrainingSetPath = [workingDirectory + "/DataSet/"+"TrainingSet_Initial"]
# Load training set
for trainingset_idx in range(len(TrainingSetPath)):
    trainingset_file = TrainingSetPath[trainingset_idx] + "/data"+'.p'
    trainingset = pickle.load(open(trainingset_file, "rb"))

    print("For dataset: ", trainingset_idx)
    print("DataSet Sizes: ")

    if trainingset_idx == 0:
        x_train = trainingset["input"]
        y_train = trainingset["output"]
    else:
        x_train = np.concatenate((x_train, trainingset["input"]), axis=0)
        y_train = np.concatenate((y_train, trainingset["output"]), axis=0)

# Build Neighbor model
# Finding the Neighbours
nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute',
                        radius=0.05).fit(x_train)

# Print Error Mesages
UnseenRoundIdx = UnseenState_data["UnseenRoundIdx"]
Recompute_Opt_Result = Recompute_data["SingleOptResultSavings"]
GroundTruth_Opt_Result = GroundTruth_data["SingleOptResultSavings"]

Shift_World_Frame_Type = "StanceFoot"
Contact_Representation_Type = "FollowRectangelBorder"

for roundIdx in range(10):

    print("Round ", UnseenRoundIdx+roundIdx, "in the GroundTruth")
    x_recompute, y_recompute = getDataPoints(SingleOptRes=Recompute_Opt_Result[roundIdx], Shift_World_Frame=Shift_World_Frame_Type,
                                             ContactRepresentationType=Contact_Representation_Type, VectorScaleFactor=1.0)  # InitCoM; InitSurfBorder

    # Brute Force
    x_query_full = np.tile(x_recompute, (x_train.shape[0], 1))
    diff = x_query_full - x_train
    distance = np.linalg.norm(diff, axis=1)
    dist_min = np.min(distance)
    dist_min_idx = np.argmin(distance)

    #distances, indices = nbrs.kneighbors(np.array([x_recompute]))

    x_ground_truth, y_ground_truth = getDataPoints(SingleOptRes=GroundTruth_Opt_Result[UnseenRoundIdx+roundIdx], Shift_World_Frame=Shift_World_Frame_Type,
                                                   ContactRepresentationType=Contact_Representation_Type, VectorScaleFactor=1.0)  # InitCoM; InitSurfBorder
    #print("Difference: ", np.linalg.norm(x_recompute - x_ground_truth))
    print("Differenc (Neighbor): ", np.linalg.norm(x_recompute -
          x_train[dist_min_idx, :]), "; Agains Ground Truth: ", np.linalg.norm(x_recompute - x_ground_truth))

# Show Figures
fig = plt.figure()
ax = Axes3D(fig)
viz.DisplayResults_Not_Show(TerrainModel=Recompute_data["TerrainInfo"], SingleOptResult=None,
                            AllOptResult=Recompute_data["SingleOptResultSavings"], fig=fig, ax=ax)
viz.DisplayResults_Not_Show(TerrainModel=GroundTruth_data["TerrainInfo"], SingleOptResult=None,
                            AllOptResult=GroundTruth_data["SingleOptResultSavings"], fig=fig, ax=ax)

plt.show()
