# Train GPR model
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
import GPy
import GPy.models as models

dataset_file = "/home/jiayu/Desktop/multicontact_learning_local_objectives/data/antfarm_2steps/DataSet/data.p"

dataset = pickle.load(open(dataset_file, "rb"))

# No Test data
# x_train, x_test, y_train, y_test = train_test_split(dataset["input"], dataset["output"])#, test_size = 0.0)
x_train = dataset["input"]
y_train = dataset["output"]

# Define gpr
gpr = models.GPRegression(x_train, y_train)

# train gpr
res = gpr.optimize_restarts(num_restarts=2)

# Predict Training data
y_pred_train, cov_test = gpr.predict(np.reshape(x_train[0], (1, 64)))

#print(np.array([x_train[0]]) - np.reshape(x_train[0], (1, 64)))

print("predicted value: ", y_pred_train)
print("true value: ", y_train[0])
