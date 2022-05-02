import numpy as np
import pickle

#Define the traj path
#traj_path = "/home/jiayu/Desktop/MLP_DataSet/TempRollOuts/rubbles_with_largeslope.p"

# traj_path = "/home/jiayu/Desktop/MLP_DataSet/TempRollOuts/largeslope.p"
traj_path = "/home/jiayu/Desktop/MLP_DataSet/TempRollOuts/temp1640016056.434527.p"

with open(traj_path, 'rb') as f:
    data = pickle.load(f)

SingleOptResult = data["SingleOptResultSavings"][10]

var_idx_lv1 = SingleOptResult["var_idx"]["Level1_Var_Index"]
x_opt = SingleOptResult["opt_res"]

print("Result of the First Level")
x_res_lv1       = np.array(x_opt[var_idx_lv1["x"][0]:var_idx_lv1["x"][1]+1]);       print('x_res: ', x_res_lv1)
y_res_lv1       = np.array(x_opt[var_idx_lv1["y"][0]:var_idx_lv1["y"][1]+1]);       print('y_res: ', y_res_lv1)
z_res_lv1       = np.array(x_opt[var_idx_lv1["z"][0]:var_idx_lv1["z"][1]+1]);       print('z_res: ', z_res_lv1)
xdot_res_lv1    = np.array(x_opt[var_idx_lv1["xdot"][0]:var_idx_lv1["xdot"][1]+1]); print('xdot_res: ', xdot_res_lv1)
ydot_res_lv1    = np.array(x_opt[var_idx_lv1["ydot"][0]:var_idx_lv1["ydot"][1]+1]); print('ydot_res: ', ydot_res_lv1)
zdot_res_lv1    = np.array(x_opt[var_idx_lv1["zdot"][0]:var_idx_lv1["zdot"][1]+1]); print('zdot_res: ', zdot_res_lv1)
Lx_res_lv1      = np.array(x_opt[var_idx_lv1["Lx"][0]:var_idx_lv1["Lx"][1]+1]);     print('Lx_res: ', Lx_res_lv1)
Ly_res_lv1      = np.array(x_opt[var_idx_lv1["Ly"][0]:var_idx_lv1["Ly"][1]+1]);     print('Ly_res: ', Ly_res_lv1)
Lz_res_lv1      = np.array(x_opt[var_idx_lv1["Lz"][0]:var_idx_lv1["Lz"][1]+1]);     print('Lz_res: ', Lz_res_lv1)
Ldotx_res_lv1      = np.array(x_opt[var_idx_lv1["Ldotx"][0]:var_idx_lv1["Ldotx"][1]+1]);     print('Ldotx_res: ', Ldotx_res_lv1)
Ldoty_res_lv1      = np.array(x_opt[var_idx_lv1["Ldoty"][0]:var_idx_lv1["Ldoty"][1]+1]);     print('Ldoty_res: ', Ldoty_res_lv1)
Ldotz_res_lv1      = np.array(x_opt[var_idx_lv1["Ldotz"][0]:var_idx_lv1["Ldotz"][1]+1]);     print('Ldotz_res: ', Ldotz_res_lv1)
px_res_lv1      = np.array(x_opt[var_idx_lv1["px"][0]:var_idx_lv1["px"][1]+1]);     print('px_res: ', px_res_lv1)
py_res_lv1      = np.array(x_opt[var_idx_lv1["py"][0]:var_idx_lv1["py"][1]+1]);     print('py_res: ', py_res_lv1)
pz_res_lv1      = np.array(x_opt[var_idx_lv1["pz"][0]:var_idx_lv1["pz"][1]+1]);     print('pz_res: ', pz_res_lv1)
Ts_res_lv1      = np.array(x_opt[var_idx_lv1["Ts"][0]:var_idx_lv1["Ts"][1]+1]);     print('Ts_res: ', Ts_res_lv1)

FL1x_res_lv1      = np.array(x_opt[var_idx_lv1["FL1x"][0]:var_idx_lv1["FL1x"][1]+1]);     print('FL1x_res: ', FL1x_res_lv1)
FL1y_res_lv1      = np.array(x_opt[var_idx_lv1["FL1y"][0]:var_idx_lv1["FL1y"][1]+1]);     print('FL1y_res: ', FL1y_res_lv1)
FL1z_res_lv1      = np.array(x_opt[var_idx_lv1["FL1z"][0]:var_idx_lv1["FL1z"][1]+1]);     print('FL1z_res: ', FL1z_res_lv1)
FL2x_res_lv1      = np.array(x_opt[var_idx_lv1["FL2x"][0]:var_idx_lv1["FL2x"][1]+1]);     print('FL2x_res: ', FL1x_res_lv1)
FL2y_res_lv1      = np.array(x_opt[var_idx_lv1["FL2y"][0]:var_idx_lv1["FL2y"][1]+1]);     print('FL2y_res: ', FL1y_res_lv1)
FL2z_res_lv1      = np.array(x_opt[var_idx_lv1["FL2z"][0]:var_idx_lv1["FL2z"][1]+1]);     print('FL2z_res: ', FL1z_res_lv1)
FL3x_res_lv1      = np.array(x_opt[var_idx_lv1["FL3x"][0]:var_idx_lv1["FL3x"][1]+1]);     print('FL3x_res: ', FL1x_res_lv1)
FL3y_res_lv1      = np.array(x_opt[var_idx_lv1["FL3y"][0]:var_idx_lv1["FL3y"][1]+1]);     print('FL3y_res: ', FL1y_res_lv1)
FL3z_res_lv1      = np.array(x_opt[var_idx_lv1["FL3z"][0]:var_idx_lv1["FL3z"][1]+1]);     print('FL3z_res: ', FL1z_res_lv1)
FL4x_res_lv1      = np.array(x_opt[var_idx_lv1["FL4x"][0]:var_idx_lv1["FL4x"][1]+1]);     print('FL4x_res: ', FL1x_res_lv1)
FL4y_res_lv1      = np.array(x_opt[var_idx_lv1["FL4y"][0]:var_idx_lv1["FL4y"][1]+1]);     print('FL4y_res: ', FL1y_res_lv1)
FL4z_res_lv1      = np.array(x_opt[var_idx_lv1["FL4z"][0]:var_idx_lv1["FL4z"][1]+1]);     print('FL4z_res: ', FL1z_res_lv1)

FR1x_res_lv1      = np.array(x_opt[var_idx_lv1["FR1x"][0]:var_idx_lv1["FR1x"][1]+1]);     print('FR1x_res: ', FR1x_res_lv1)
FR1y_res_lv1      = np.array(x_opt[var_idx_lv1["FR1y"][0]:var_idx_lv1["FR1y"][1]+1]);     print('FR1y_res: ', FR1y_res_lv1)
FR1z_res_lv1      = np.array(x_opt[var_idx_lv1["FR1z"][0]:var_idx_lv1["FR1z"][1]+1]);     print('FR1z_res: ', FR1z_res_lv1)
FR2x_res_lv1      = np.array(x_opt[var_idx_lv1["FR2x"][0]:var_idx_lv1["FR2x"][1]+1]);     print('FR2x_res: ', FR1x_res_lv1)
FR2y_res_lv1      = np.array(x_opt[var_idx_lv1["FR2y"][0]:var_idx_lv1["FR2y"][1]+1]);     print('FR2y_res: ', FR1y_res_lv1)
FR2z_res_lv1      = np.array(x_opt[var_idx_lv1["FR2z"][0]:var_idx_lv1["FR2z"][1]+1]);     print('FR2z_res: ', FR1z_res_lv1)
FR3x_res_lv1      = np.array(x_opt[var_idx_lv1["FR3x"][0]:var_idx_lv1["FR3x"][1]+1]);     print('FR3x_res: ', FR1x_res_lv1)
FR3y_res_lv1      = np.array(x_opt[var_idx_lv1["FR3y"][0]:var_idx_lv1["FR3y"][1]+1]);     print('FR3y_res: ', FR1y_res_lv1)
FR3z_res_lv1      = np.array(x_opt[var_idx_lv1["FR3z"][0]:var_idx_lv1["FR3z"][1]+1]);     print('FR3z_res: ', FR1z_res_lv1)
FR4x_res_lv1      = np.array(x_opt[var_idx_lv1["FR4x"][0]:var_idx_lv1["FR4x"][1]+1]);     print('FR4x_res: ', FR1x_res_lv1)
FR4y_res_lv1      = np.array(x_opt[var_idx_lv1["FR4y"][0]:var_idx_lv1["FR4y"][1]+1]);     print('FR4y_res: ', FR1y_res_lv1)
FR4z_res_lv1      = np.array(x_opt[var_idx_lv1["FR4z"][0]:var_idx_lv1["FR4z"][1]+1]);     print('FR4z_res: ', FR1z_res_lv1)

PL_init_TangentX = SingleOptResult["PL_init_TangentX"]
PL_init_TangentY = SingleOptResult["PL_init_TangentY"]
PL_init_Norm = SingleOptResult["PL_init_Norm"]
print("PL_init_TangentX: ", PL_init_TangentX)
print("PL_init_TangentY: ", PL_init_TangentY)
print("PL_init_Norm: ", PL_init_Norm)

PR_init_TangentX = SingleOptResult["PR_init_TangentX"]
PR_init_TangentY = SingleOptResult["PR_init_TangentY"]
PR_init_Norm = SingleOptResult["PR_init_Norm"]
print("PR_init_TangentX: ", PR_init_TangentX)
print("PR_init_TangentY: ", PR_init_TangentY)
print("PR_init_Norm: ", PR_init_Norm)

Contact_TangentX = SingleOptResult["SurfTangentsX"][0]
Contact_TangentY = SingleOptResult["SurfTangentsY"][0]
Contact_Norm = SingleOptResult["SurfNorms"][0]
print("Contact_TangentX: ", Contact_TangentX)
print("Contact_TangentY: ", Contact_TangentY)
print("Contact_Norm: ", Contact_Norm)

PL_init = np.array([SingleOptResult["PLx_init"], SingleOptResult["PLy_init"], SingleOptResult["PLz_init"]])
print("Init Left Contact Location: ", PL_init)
PR_init = np.array([SingleOptResult["PRx_init"], SingleOptResult["PRy_init"], SingleOptResult["PRz_init"]])
print("Init Right Contact Location: ", PR_init)

#--------
knotIdx = 23
#--------

CoM = np.array([x_res_lv1[knotIdx], y_res_lv1[knotIdx], z_res_lv1[knotIdx]])
Ldot = np.array([Ldotx_res_lv1[knotIdx], Ldoty_res_lv1[knotIdx], Ldotz_res_lv1[knotIdx]])

P = np.array([px_res_lv1[0],py_res_lv1[0],pz_res_lv1[0]])
print("P: ", P)

FL1 = np.array([FL1x_res_lv1[knotIdx],FL1y_res_lv1[knotIdx],FL1z_res_lv1[knotIdx]])
FL2 = np.array([FL2x_res_lv1[knotIdx],FL2y_res_lv1[knotIdx],FL2z_res_lv1[knotIdx]])
FL3 = np.array([FL3x_res_lv1[knotIdx],FL3y_res_lv1[knotIdx],FL3z_res_lv1[knotIdx]])
FL4 = np.array([FL4x_res_lv1[knotIdx],FL4y_res_lv1[knotIdx],FL4z_res_lv1[knotIdx]])

FR1 = np.array([FR1x_res_lv1[knotIdx],FR1y_res_lv1[knotIdx],FR1z_res_lv1[knotIdx]])
FR2 = np.array([FR2x_res_lv1[knotIdx],FR2y_res_lv1[knotIdx],FR2z_res_lv1[knotIdx]])
FR3 = np.array([FR3x_res_lv1[knotIdx],FR3y_res_lv1[knotIdx],FR3z_res_lv1[knotIdx]])
FR4 = np.array([FR4x_res_lv1[knotIdx],FR4y_res_lv1[knotIdx],FR4z_res_lv1[knotIdx]])

#Init double suppport 

if knotIdx >= 0 and knotIdx <= 7:
    Ldot_compute = np.cross(PL_init+0.11*PL_init_TangentX+0.06*PL_init_TangentY-CoM,FL1) + \
                np.cross(PL_init+0.11*PL_init_TangentX-0.06*PL_init_TangentY-CoM,FL2) + \
                np.cross(PL_init-0.11*PL_init_TangentX+0.06*PL_init_TangentY-CoM,FL3) + \
                np.cross(PL_init-0.11*PL_init_TangentX-0.06*PL_init_TangentY-CoM,FL4) + \
                np.cross(PR_init+0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM,FR1) + \
                np.cross(PR_init+0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM,FR2) + \
                np.cross(PR_init-0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM,FR3) + \
                np.cross(PR_init-0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM,FR4)

# #Swing Phase
if knotIdx >= 8 and knotIdx <= 15:
    Ldot_compute = np.cross(PR_init +0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM,FR1) + \
                np.cross(PR_init +0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM,FR2) + \
                np.cross(PR_init -0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM,FR3) + \
                np.cross(PR_init -0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM,FR4) 

#Double Support Phase
if knotIdx >= 16 and knotIdx <= 23:
    Ldot_compute = np.cross(PR_init +0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM,FR1) + \
                np.cross(PR_init +0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM,FR2) + \
                np.cross(PR_init -0.11*PR_init_TangentX+0.06*PR_init_TangentY-CoM,FR3) + \
                np.cross(PR_init -0.11*PR_init_TangentX-0.06*PR_init_TangentY-CoM,FR4) + \
                np.cross(P +0.11*Contact_TangentX+0.06*Contact_TangentY-CoM,FL1) + \
                np.cross(P +0.11*Contact_TangentX-0.06*Contact_TangentY-CoM,FL2) + \
                np.cross(P -0.11*Contact_TangentX+0.06*Contact_TangentY-CoM,FL3) + \
                np.cross(P -0.11*Contact_TangentX-0.06*Contact_TangentY-CoM,FL4)

print("Original Ldot: ", Ldot)
print("Computed Ldot: ", Ldot_compute)
