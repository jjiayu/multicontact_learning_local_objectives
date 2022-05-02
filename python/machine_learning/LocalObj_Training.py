# Import Packages
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle
from multicontact_learning_local_objectives.python.machine_learning.ml_utils import *
import matplotlib.pyplot as plt  # Matplotlib
import time
import shutil
import sys

# --------------------------
# Define Path for Storing Trajectories
# Collect Data Points Path
#workingDirectory = "/home/jiayu/Desktop/multicontact_learning_local_objectives/data/large_slope_flat_patches/"
#workingDirectory = "/afs/inf.ed.ac.uk/group/project/mlp_localobj/Rubbles_and_OneLargeSlope/"

# Get Working Directory and Number of Epochs from Parameters
workingDirectory = sys.argv[1]
print("Working Folder: \n", workingDirectory)
# Get Number of Epochs
if len(sys.argv) < 3:  # if not defined, then give default number
    NumEpochs = 25000
else:
    NumEpochs = int(sys.argv[2])
print("Number of Training Epochs: ", NumEpochs)

# --------------------------
# Define Log Saving Path and start logging
stdoutOrigin = sys.stdout
sys.stdout = open(workingDirectory+"/training_log.txt", "w")
# Repeat Printing for Logging
print("Working Folder: \n", workingDirectory)
print("Number of Training Epochs: ", NumEpochs)

# ------------------------
DataSetPath = workingDirectory + "/DataSet"
# Path to store ML Model
ML_Model_Path = workingDirectory + "/ML_Models/"
# Clean the old data, delete the folder
if os.path.isdir(ML_Model_Path):
    shutil.rmtree(ML_Model_Path)
# Make the folder for the DataSet
os.mkdir(ML_Model_Path)

# Give some time to show the working directory
time.sleep(7)

# ---------------------------------------------------------
# Train Neural Network


# Load Datta set
# Dataset filename
dataset_file = DataSetPath + "/data"+'.p'
dataset = pickle.load(open(dataset_file, "rb"))

print("World Frame Shift: ", dataset["Shift_World_Frame_Type"])
print("Contact Location Representation Type: ",
      dataset["Contact_Representation_Type"])
print("Scaling Factor of Variables: ", dataset["VectorScaleFactor"])
print("Number of Preview Steps: ", dataset["NumPreviewSteps"])

# Test Train Split, for now No Test data
#x_train, x_test, y_train, y_test = train_test_split(dataset["input"], dataset["output"], test_size = 0.01)
x_train = dataset["input"]
y_train = dataset["output"]

# Decide input and outpu dimensionality
d_in = x_train[0].shape[0]
d_out = y_train[0].shape[0]

# Define learning model
model = Sequential([
    Dense(256, activation='relu', input_shape=(d_in,)),
    Dense(256, activation='relu', input_shape=(d_in,)),
    Dense(256, activation='relu', input_shape=(d_in,)),
    Dense(256, activation='relu', input_shape=(d_in,)),
    Dense(d_out)
])

# Train Learning Model
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='mse')

#history = model.fit(x_train, y_train, epochs = 50000, validation_split=0.0, batch_size = x_train.shape[0])
history = model.fit(x_train, y_train, epochs=NumEpochs, validation_split=0.2)

# Save Trained Model
MLmodel_name = "NN_Model_Valid"
model.save(ML_Model_Path + MLmodel_name)

# Save DataSet Setttings
datasetSettings = {"Shift_World_Frame_Type": dataset["Shift_World_Frame_Type"],
                   "VectorScaleFactor": dataset["VectorScaleFactor"],
                   "NumPreviewSteps": dataset["NumPreviewSteps"],
                   "Contact_Representation_Type": dataset["Contact_Representation_Type"],
                   "TrainingLoss": history.history['loss'],
                   "ValidationLoss": history.history['val_loss']}
pickle.dump(datasetSettings, open(ML_Model_Path +
            MLmodel_name + '/datasetSettings' + '.p', "wb"))
