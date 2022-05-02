import numpy as np
import matplotlib.pyplot as plt  # Matplotlib
import multicontact_learning_local_objectives.python.visualization as viz
import pickle

fig = plt.figure()
ax = plt.axes()

recompute_traj = "/home/jiayu/Desktop/MLP_DataSet/Rubbles_RegretOneStep/RollOuts_from_Unseen_1StepBeforeFail_TrackTraining_Aug_1StepBeforeFail_2Time/Group1_temp1633949865.1425622.p"

datatemp = pickle.load((open(recompute_traj, "rb")))

UnseenStateFilePath = datatemp["TerrainModelPath"]
idxTemp = UnseenStateFilePath.find("Rubbles_RegretOneStep")
UnseenStateFilePath = "/home/jiayu/Desktop/MLP_DataSet/" + \
    UnseenStateFilePath[idxTemp:]

datatemp = pickle.load((open(UnseenStateFilePath, "rb")))
RollOutTrackingPath = datatemp["TrackingExpPath"]

idxTemp = RollOutTrackingPath.find("Rubbles_RegretOneStep")
RollOutTrackingPath = "/home/jiayu/Desktop/MLP_DataSet/" + \
    RollOutTrackingPath[idxTemp:]
datatemp = pickle.load((open(RollOutTrackingPath, "rb")))

NLP_traj_Path = datatemp["TerrainModelPath"]

idxTemp = NLP_traj_Path.find("Rubbles_RegretOneStep")
NLP_traj_Path = "/home/jiayu/Desktop/MLP_DataSet/"+NLP_traj_Path[idxTemp:]


print(NLP_traj_Path)

query_traj = "y_result"

viz.draw_timeSeries_and_traj(filePath=recompute_traj, query_traj=query_traj,
                             traj_color=None, startStepNum=0, EndStepNum=29, NumLocalKonts=8, fig=fig, ax=ax)
viz.draw_timeSeries_and_traj(filePath=NLP_traj_Path, query_traj=query_traj,
                             traj_color=None, startStepNum=0, EndStepNum=29, NumLocalKonts=8, fig=fig, ax=ax)

plt.show()
