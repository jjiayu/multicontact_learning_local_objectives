#For computing Kinematics Polytopes
from sl1m.problem_definition import *
from sl1m.planner_scenarios.talos.constraints_shift import *

import numpy as np
import pickle

#Get Kinematics Constraint for Talos
#CoM kinematics constraint, give homogenous transformaiton (the last column seems like dont make a diff)
K_CoM_Right,k_CoM_Right = right_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
K_CoM_Left,k_CoM_Left = left_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))

#Relative Foot Constraint matrices
#Relative foot constraint, give homogenous transformation (the last column seems like dont make a diff)
Q_rf_in_lf,q_rf_in_lf = right_foot_in_lf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
Q_lf_in_rf,q_lf_in_rf = left_foot_in_rf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))

Kinematcis_Polytope = {"K_CoM_in_Right_Contact": K_CoM_Right,    "k_CoM_in_Right_Contact": k_CoM_Right,
                       "K_CoM_in_Left_Contact" : K_CoM_Left,      "k_CoM_in_Left_Contact" : k_CoM_Left,
                       "Q_Right_Contact_in_Left_Contact": Q_rf_in_lf,   "q_Right_Contact_in_Left_Contact": q_rf_in_lf,
                       "Q_Left_Contact_in_Right_Contact": Q_lf_in_rf,   "q_Left_Contact_in_Right_Contact": q_lf_in_rf}

pickle.dump(Kinematcis_Polytope, open("kinematics_constraints"+".p", "wb"))    #Save Data
pickle.dump(Kinematcis_Polytope, open("../ocp_build/kinematics_constraints"+".p", "wb"))    #Save Data in other places
pickle.dump(Kinematcis_Polytope, open("../rhp_plan/kinematics_constraints"+".p", "wb"))    #Save Data in other places

with open("kinematics_constraints.p", 'rb') as f:
    kinematics_constraints= pickle.load(f)

print(kinematics_constraints)

