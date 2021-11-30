#Tools for processing Trajectories

import numpy as np
from .RefFrameTrans import *

#get initial and terminal condition from a single trajectory optimization round; for the execution horizon
#NOTE: The InitConfig here is a small subset of the InitConfig comparing to the one in rhp_gen.py, because it is just used to get local frame transformations
#      or collect for data points for machine learning
def getInitTerminalConfig(SingleOptRes = None, Shift_World_Frame = None):
    
    #Make Init and Terminal Config Containers
    InitConfig = {};     TerminalConfig = {}

    #------------------------
    #Get Var index of the first level
    var_idx_lv1 = SingleOptRes["var_idx"]["Level1_Var_Index"]

    #Get Full optimization result, NOTE: Quantities Multiplied with Scaling Factor as well
    x_opt = SingleOptRes["opt_res"]

    #Get CoM x, y, z init, and terminal
    x_lv1_res = np.array(x_opt[var_idx_lv1["x"][0]:var_idx_lv1["x"][1]+1]);       InitConfig["x_init"] = x_lv1_res[0];    TerminalConfig["x_end"] = x_lv1_res[-1]
    y_lv1_res = np.array(x_opt[var_idx_lv1["y"][0]:var_idx_lv1["y"][1]+1]);       InitConfig["y_init"] = y_lv1_res[0];    TerminalConfig["y_end"] = y_lv1_res[-1]
    z_lv1_res = np.array(x_opt[var_idx_lv1["z"][0]:var_idx_lv1["z"][1]+1]);       InitConfig["z_init"] = z_lv1_res[0];    TerminalConfig["z_end"] = z_lv1_res[-1]
    #Get CoMdot x, y, z init and terminal
    xdot_lv1_res = np.array(x_opt[var_idx_lv1["xdot"][0]:var_idx_lv1["xdot"][1]+1]);   InitConfig["xdot_init"] = xdot_lv1_res[0];   TerminalConfig["xdot_end"] = xdot_lv1_res[-1]
    ydot_lv1_res = np.array(x_opt[var_idx_lv1["ydot"][0]:var_idx_lv1["ydot"][1]+1]);   InitConfig["ydot_init"] = ydot_lv1_res[0];   TerminalConfig["ydot_end"] = ydot_lv1_res[-1]
    zdot_lv1_res = np.array(x_opt[var_idx_lv1["zdot"][0]:var_idx_lv1["zdot"][1]+1]);   InitConfig["zdot_init"] = zdot_lv1_res[0];   TerminalConfig["zdot_end"] = zdot_lv1_res[-1]
    #Get CoMdot x, y, z init and terminal
    Lx_lv1_res = np.array(x_opt[var_idx_lv1["Lx"][0]:var_idx_lv1["Lx"][1]+1]);   InitConfig["Lx_init"] = Lx_lv1_res[0];   TerminalConfig["Lx_end"] = Lx_lv1_res[-1]
    Ly_lv1_res = np.array(x_opt[var_idx_lv1["Ly"][0]:var_idx_lv1["Ly"][1]+1]);   InitConfig["Ly_init"] = Ly_lv1_res[0];   TerminalConfig["Ly_end"] = Ly_lv1_res[-1]
    Lz_lv1_res = np.array(x_opt[var_idx_lv1["Lz"][0]:var_idx_lv1["Lz"][1]+1]);   InitConfig["Lz_init"] = Lz_lv1_res[0];   TerminalConfig["Lz_end"] = Lz_lv1_res[-1]

    #Get Init Left and Right Contacts
    InitConfig["PLx_init"] = np.array(SingleOptRes["PLx_init"]);     InitConfig["PLy_init"] = np.array(SingleOptRes["PLy_init"])    
    InitConfig["PLz_init"] = np.array(SingleOptRes["PLz_init"])

    InitConfig["PRx_init"] = np.array(SingleOptRes["PRx_init"]);     InitConfig["PRy_init"] = np.array(SingleOptRes["PRy_init"])       
    InitConfig["PRz_init"] = np.array(SingleOptRes["PRz_init"])

    #Get Right and Left Init Contact Surfaces
    InitConfig["LeftInitSurf"] = SingleOptRes["LeftInitSurf"];     InitConfig["RightInitSurf"] = SingleOptRes["RightInitSurf"]

    #Get Right and Left Foot Init Tangents
    InitConfig["PL_init_TangentX"] = SingleOptRes["PL_init_TangentX"];  InitConfig["PL_init_TangentY"] = SingleOptRes["PL_init_TangentY"];  InitConfig["PL_init_Norm"] = SingleOptRes["PL_init_Norm"]
    InitConfig["PR_init_TangentX"] = SingleOptRes["PR_init_TangentX"];  InitConfig["PR_init_TangentY"] = SingleOptRes["PR_init_TangentY"];  InitConfig["PR_init_Norm"] = SingleOptRes["PR_init_Norm"]

    #Get Right and Left Init Contact Orientation
    InitConfig["LeftInitSurfOrientation"] = SingleOptRes["LeftInitSurfOrientation"]
    InitConfig["RightInitSurfOrientation"] = SingleOptRes["RightInitSurfOrientation"]

    #Get Contact Location
    TerminalConfig["Px"] = np.array(x_opt[var_idx_lv1["px"][0]:var_idx_lv1["px"][1]+1])
    TerminalConfig["Py"] = np.array(x_opt[var_idx_lv1["py"][0]:var_idx_lv1["py"][1]+1])
    TerminalConfig["Pz"] = np.array(x_opt[var_idx_lv1["pz"][0]:var_idx_lv1["pz"][1]+1])

    #Get Swing Foot Flag
    InitConfig["LeftSwingFlag"]  = SingleOptRes["LeftSwingFlag"] #We only use LeftSwingFlag to indicate the which foot is moving
    InitConfig["RightSwingFlag"] = SingleOptRes["RightSwingFlag"] #We only use LeftSwingFlag to indicate the which foot is moving

    #Get Contact Surfs, Contact Tangents
    InitConfig["ContactSurfs"] = SingleOptRes["ContactSurfs"]
    InitConfig["SurfTangentsX"] = SingleOptRes["SurfTangentsX"]
    InitConfig["SurfTangentsY"] = SingleOptRes["SurfTangentsY"]
    InitConfig["SurfOrientations"] = SingleOptRes["SurfOrientations"]

    #Shift Variables NOTE: the function copies the InitConfig we pass into, and Tangent, Norm Orientaiton are not Transformed
    shiftedInitConfig, shiftedTerminalConfig = shiftInitTerminalConfig_to_LocalFrame(InitConfig = InitConfig, TerminalConfig = TerminalConfig, Local_Frame_Selection = Shift_World_Frame)

    return shiftedInitConfig, shiftedTerminalConfig