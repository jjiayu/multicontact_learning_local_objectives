from re import I
import numpy as np
import pickle
from multicontact_learning_local_objectives.python.utils import *
import os
import copy

#--------
# #TensorFlow when use as tracking local obj
# #Use CPU for Tensorflow
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
# import tensorflow as tf
# from tensorflow.keras.models import load_model


#Set Initial Condition for the first step
#   InitConditionType = "null": null state
#   Others, get from other cases
def getInitCondition_FirstStep(InitConditionType = None, InitConditionFilePath = None):
    if InitConditionType == None:
        raise Exception("Error::Wrong Initial Condition Type0")
    elif InitConditionType == "null":
        #SwingLeg Flag
        SwingLeftFirst = 1;   SwingRightFirst = 0
        #   Initial CoM State
        x_init = 0.0;           y_init = 0.0;      z_init = 0.85 #0.75
        xdot_init = 0.0;        ydot_init = 0.0;   zdot_init = 0.0
        Lx_init = 0.0;          Ly_init = 0.0;     Lz_init = 0.0
        Ldotx_init = 0.0;       Ldoty_init = 0.0;  Ldotz_init = 0.0
        #   Initial Contact Location Left Foot
        PLx_init = x_init;    
        PLy_init = y_init + 0.17/2.0;  
        PLz_init = 0.0 #Distance between right and left foot is 0.17
        #   Initial Contact Location Right Foot
        PRx_init = x_init;    
        PRy_init = y_init - 0.17/2.0;  
        PRz_init = 0.0
        ##   Initial Contact Tagents and Norms, should get from terrain patches
        #PL_init_TangentX = np.array([1,0,0]);   PL_init_TangentY = np.array([0,1,0]);   PL_init_Norm = np.array([0,0,1])
        #PR_init_TangentX = np.array([1,0,0]);   PR_init_TangentY = np.array([0,1,0]);   PR_init_Norm = np.array([0,0,1])
        #print(" ")
    elif InitConditionType == "fromFile":
        if InitConditionFilePath == None:
            raise Exception("Unprovided InitCondition File Path")
        
        #Get Init Condition data
        with open(InitConditionFilePath, 'rb') as f:
            InitConditionData = pickle.load(f)
        
        #Unpack Init Condition
        #   Swing Foot Flag of the First Step
        SwingLeftFirst = InitConditionData["InitConfig"]["LeftSwingFlag"]
        SwingRightFirst= InitConditionData["InitConfig"]["RightSwingFlag"]

        #   Initial CoM State
        x_init = InitConditionData["InitConfig"]["x_init"]         
        y_init = InitConditionData["InitConfig"]["y_init"]     
        z_init = InitConditionData["InitConfig"]["z_init"] 
        xdot_init = InitConditionData["InitConfig"]["xdot_init"] 
        ydot_init = InitConditionData["InitConfig"]["ydot_init"] 
        zdot_init = InitConditionData["InitConfig"]["zdot_init"] 
        Lx_init = InitConditionData["InitConfig"]["Lx_init"] 
        Ly_init = InitConditionData["InitConfig"]["Ly_init"]
        Lz_init = InitConditionData["InitConfig"]["Lz_init"]
        Ldotx_init = InitConditionData["InitConfig"]["Ldotx_init"]
        Ldoty_init = InitConditionData["InitConfig"]["Ldoty_init"]
        Ldotz_init = InitConditionData["InitConfig"]["Ldotz_init"]
        #   Initial Contact Location Left Foot
        PLx_init = InitConditionData["InitConfig"]["PLx_init"]
        PLy_init = InitConditionData["InitConfig"]["PLy_init"]
        PLz_init = InitConditionData["InitConfig"]["PLz_init"]
        #   Initial Contact Location Right Foot
        PRx_init = InitConditionData["InitConfig"]["PRx_init"]
        PRy_init = InitConditionData["InitConfig"]["PRy_init"]
        PRz_init = InitConditionData["InitConfig"]["PRz_init"]
        
    elif InitConditionType == "fromFirstRoundTraj":
        if InitConditionFilePath == None:
            raise Exception("Traj File (for getting Initial Condition) is not found")

        #Get Init Condition data
        with open(InitConditionFilePath, 'rb') as f:
            InitConditionData = pickle.load(f)

        #Get Initial Conditions from the first round
        FirstRound_OptRes = InitConditionData["SingleOptResultSavings"][0]
        FirstRound_InitConfig, FirstRound_TerminalConfig = getInitConfig_in_GlobalFrame_from_SingleOptResult(SingleOptRes=FirstRound_OptRes)

        #Unpack Init Condition
        #   Swing Foot Flag of the First Step
        SwingLeftFirst = FirstRound_InitConfig["LeftSwingFlag"]
        SwingRightFirst= FirstRound_InitConfig["RightSwingFlag"]

        #   Initial CoM State
        x_init = FirstRound_InitConfig["x_init"]         
        y_init = FirstRound_InitConfig["y_init"]     
        z_init = FirstRound_InitConfig["z_init"] 
        xdot_init = FirstRound_InitConfig["xdot_init"] 
        ydot_init = FirstRound_InitConfig["ydot_init"] 
        zdot_init = FirstRound_InitConfig["zdot_init"] 
        Lx_init = FirstRound_InitConfig["Lx_init"] 
        Ly_init = FirstRound_InitConfig["Ly_init"]
        Lz_init = FirstRound_InitConfig["Lz_init"]
        Ldotx_init = FirstRound_InitConfig["Ldotx_init"]
        Ldoty_init = FirstRound_InitConfig["Ldoty_init"]
        Ldotz_init = FirstRound_InitConfig["Ldotz_init"]
        #   Initial Contact Location Left Foot
        PLx_init = FirstRound_InitConfig["PLx_init"]
        PLy_init = FirstRound_InitConfig["PLy_init"]
        PLz_init = FirstRound_InitConfig["PLz_init"]
        #   Initial Contact Location Right Foot
        PRx_init = FirstRound_InitConfig["PRx_init"]
        PRy_init = FirstRound_InitConfig["PRy_init"]
        PRz_init = FirstRound_InitConfig["PRz_init"]
        
    else:
        raise Exception("Unknown Initial Condition Type")

    return SwingLeftFirst,  SwingRightFirst, \
           x_init,       y_init,       z_init,     \
           xdot_init,    ydot_init,    zdot_init,  \
           Lx_init,      Ly_init,      Lz_init,    \
           Ldotx_init,   Ldoty_init,   Ldotz_init, \
           PLx_init,     PLy_init,     PLz_init,   \
           PRx_init,     PRy_init,     PRz_init


#Extract GroundTruch Intial and Terminal Condition From file
def getInitConfig_in_GlobalFrame_from_file(FilePath=None, RoundNum=None):
    
    #Load data
    with open(FilePath, 'rb') as f:
        data= pickle.load(f)

    #Get Single Opt Result
    SingleOptRes=data["SingleOptResultSavings"][RoundNum]

    #Make Init and Terminal Config Containers
    InitConfig = {}; TerminalConfig = {}

    #------------------------
    #Get Var index of the first level
    var_idx_lv1 = SingleOptRes["var_idx"]["Level1_Var_Index"]

    #Get Full optimization result, NOTE: Quantities Multiplied with Scaling Factor as well
    x_opt = SingleOptRes["opt_res"]

    #Get CoM x, y, z init, and terminal
    x_lv1_res = np.array(x_opt[var_idx_lv1["x"][0]:var_idx_lv1["x"][1]+1]);       InitConfig["x_init"] = x_lv1_res[0];   TerminalConfig["x_end"] = x_lv1_res[-1]
    y_lv1_res = np.array(x_opt[var_idx_lv1["y"][0]:var_idx_lv1["y"][1]+1]);       InitConfig["y_init"] = y_lv1_res[0];   TerminalConfig["y_end"] = y_lv1_res[-1]
    z_lv1_res = np.array(x_opt[var_idx_lv1["z"][0]:var_idx_lv1["z"][1]+1]);       InitConfig["z_init"] = z_lv1_res[0];   TerminalConfig["z_end"] = z_lv1_res[-1]
    #Get CoMdot x, y, z init and terminal
    xdot_lv1_res = np.array(x_opt[var_idx_lv1["xdot"][0]:var_idx_lv1["xdot"][1]+1]);   InitConfig["xdot_init"] = xdot_lv1_res[0];   TerminalConfig["xdot_end"] = xdot_lv1_res[-1]
    ydot_lv1_res = np.array(x_opt[var_idx_lv1["ydot"][0]:var_idx_lv1["ydot"][1]+1]);   InitConfig["ydot_init"] = ydot_lv1_res[0];   TerminalConfig["ydot_end"] = ydot_lv1_res[-1]
    zdot_lv1_res = np.array(x_opt[var_idx_lv1["zdot"][0]:var_idx_lv1["zdot"][1]+1]);   InitConfig["zdot_init"] = zdot_lv1_res[0];   TerminalConfig["zdot_end"] = zdot_lv1_res[-1]
    #Get CoMdot x, y, z init and terminal
    Lx_lv1_res = np.array(x_opt[var_idx_lv1["Lx"][0]:var_idx_lv1["Lx"][1]+1]);   InitConfig["Lx_init"] = Lx_lv1_res[0];   TerminalConfig["Lx_end"] = Lx_lv1_res[-1]
    Ly_lv1_res = np.array(x_opt[var_idx_lv1["Ly"][0]:var_idx_lv1["Ly"][1]+1]);   InitConfig["Ly_init"] = Ly_lv1_res[0];   TerminalConfig["Ly_end"] = Ly_lv1_res[-1]
    Lz_lv1_res = np.array(x_opt[var_idx_lv1["Lz"][0]:var_idx_lv1["Lz"][1]+1]);   InitConfig["Lz_init"] = Lz_lv1_res[0];   TerminalConfig["Lz_end"] = Lz_lv1_res[-1]

    #Get Ldot init and terminal
    Ldotx_lv1_res = np.array(x_opt[var_idx_lv1["Ldotx"][0]:var_idx_lv1["Ldotx"][1]+1]);   InitConfig["Ldotx_init"] = Ldotx_lv1_res[0];   TerminalConfig["Ldotx_end"] = Ldotx_lv1_res[-1]
    Ldoty_lv1_res = np.array(x_opt[var_idx_lv1["Ldoty"][0]:var_idx_lv1["Ldoty"][1]+1]);   InitConfig["Ldoty_init"] = Ldoty_lv1_res[0];   TerminalConfig["Ldoty_end"] = Ldoty_lv1_res[-1]
    Ldotz_lv1_res = np.array(x_opt[var_idx_lv1["Ldotz"][0]:var_idx_lv1["Ldotz"][1]+1]);   InitConfig["Ldotz_init"] = Ldotz_lv1_res[0];   TerminalConfig["Ldotz_end"] = Ldotz_lv1_res[-1]

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

    #Get Swing Foot Flag
    InitConfig["LeftSwingFlag"]  = SingleOptRes["LeftSwingFlag"] #We only use LeftSwingFlag to indicate the which foot is moving
    InitConfig["RightSwingFlag"] = SingleOptRes["RightSwingFlag"] #We only use LeftSwingFlag to indicate the which foot is moving

    #Get Contact Surfs, Contact Tangents
    InitConfig["ContactSurfs"] = SingleOptRes["ContactSurfs"]
    InitConfig["SurfTangentsX"] = SingleOptRes["SurfTangentsX"]
    InitConfig["SurfTangentsY"] = SingleOptRes["SurfTangentsY"]
    InitConfig["SurfOrientations"] = SingleOptRes["SurfOrientations"]

    #Get Contact Locations
    TerminalConfig["Px"] = np.array(x_opt[var_idx_lv1["px"][0]:var_idx_lv1["px"][1]+1])
    TerminalConfig["Py"] = np.array(x_opt[var_idx_lv1["py"][0]:var_idx_lv1["py"][1]+1])
    TerminalConfig["Pz"] = np.array(x_opt[var_idx_lv1["pz"][0]:var_idx_lv1["pz"][1]+1])

    return InitConfig, TerminalConfig


#Extract GroundTruch Intial and Terminal Condition From file
def getInitConfig_in_GlobalFrame_from_SingleOptResult(SingleOptRes = None):
    
    if SingleOptRes == None:
        raise Exception("Unprovided Single Opt Result")

    #Get Single Opt Result
    #SingleOptRes=data["SingleOptResultSavings"][RoundNum]

    #Make Init and Terminal Config Containers
    InitConfig = {}; TerminalConfig = {}

    #------------------------
    #Get Var index of the first level
    var_idx_lv1 = SingleOptRes["var_idx"]["Level1_Var_Index"]

    #Get Full optimization result, NOTE: Quantities Multiplied with Scaling Factor as well
    x_opt = SingleOptRes["opt_res"]

    #Get CoM x, y, z init, and terminal
    x_lv1_res = np.array(x_opt[var_idx_lv1["x"][0]:var_idx_lv1["x"][1]+1]);       InitConfig["x_init"] = x_lv1_res[0];   TerminalConfig["x_end"] = x_lv1_res[-1]
    y_lv1_res = np.array(x_opt[var_idx_lv1["y"][0]:var_idx_lv1["y"][1]+1]);       InitConfig["y_init"] = y_lv1_res[0];   TerminalConfig["y_end"] = y_lv1_res[-1]
    z_lv1_res = np.array(x_opt[var_idx_lv1["z"][0]:var_idx_lv1["z"][1]+1]);       InitConfig["z_init"] = z_lv1_res[0];   TerminalConfig["z_end"] = z_lv1_res[-1]
    #Get CoMdot x, y, z init and terminal
    xdot_lv1_res = np.array(x_opt[var_idx_lv1["xdot"][0]:var_idx_lv1["xdot"][1]+1]);   InitConfig["xdot_init"] = xdot_lv1_res[0];   TerminalConfig["xdot_end"] = xdot_lv1_res[-1]
    ydot_lv1_res = np.array(x_opt[var_idx_lv1["ydot"][0]:var_idx_lv1["ydot"][1]+1]);   InitConfig["ydot_init"] = ydot_lv1_res[0];   TerminalConfig["ydot_end"] = ydot_lv1_res[-1]
    zdot_lv1_res = np.array(x_opt[var_idx_lv1["zdot"][0]:var_idx_lv1["zdot"][1]+1]);   InitConfig["zdot_init"] = zdot_lv1_res[0];   TerminalConfig["zdot_end"] = zdot_lv1_res[-1]
    #Get L init and terminal
    Lx_lv1_res = np.array(x_opt[var_idx_lv1["Lx"][0]:var_idx_lv1["Lx"][1]+1]);   InitConfig["Lx_init"] = Lx_lv1_res[0];   TerminalConfig["Lx_end"] = Lx_lv1_res[-1]
    Ly_lv1_res = np.array(x_opt[var_idx_lv1["Ly"][0]:var_idx_lv1["Ly"][1]+1]);   InitConfig["Ly_init"] = Ly_lv1_res[0];   TerminalConfig["Ly_end"] = Ly_lv1_res[-1]
    Lz_lv1_res = np.array(x_opt[var_idx_lv1["Lz"][0]:var_idx_lv1["Lz"][1]+1]);   InitConfig["Lz_init"] = Lz_lv1_res[0];   TerminalConfig["Lz_end"] = Lz_lv1_res[-1]
    #Get Ldot init and terminal
    Ldotx_lv1_res = np.array(x_opt[var_idx_lv1["Ldotx"][0]:var_idx_lv1["Ldotx"][1]+1]);   InitConfig["Ldotx_init"] = Ldotx_lv1_res[0];   TerminalConfig["Ldotx_end"] = Ldotx_lv1_res[-1]
    Ldoty_lv1_res = np.array(x_opt[var_idx_lv1["Ldoty"][0]:var_idx_lv1["Ldoty"][1]+1]);   InitConfig["Ldoty_init"] = Ldoty_lv1_res[0];   TerminalConfig["Ldoty_end"] = Ldoty_lv1_res[-1]
    Ldotz_lv1_res = np.array(x_opt[var_idx_lv1["Ldotz"][0]:var_idx_lv1["Ldotz"][1]+1]);   InitConfig["Ldotz_init"] = Ldotz_lv1_res[0];   TerminalConfig["Ldotz_end"] = Ldotz_lv1_res[-1]

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

    #Get Swing Foot Flag
    InitConfig["LeftSwingFlag"]  = SingleOptRes["LeftSwingFlag"] #We only use LeftSwingFlag to indicate the which foot is moving
    InitConfig["RightSwingFlag"] = SingleOptRes["RightSwingFlag"] #We only use LeftSwingFlag to indicate the which foot is moving

    #Get Contact Surfs, Contact Tangents
    InitConfig["ContactSurfs"] = SingleOptRes["ContactSurfs"]
    InitConfig["SurfTangentsX"] = SingleOptRes["SurfTangentsX"]
    InitConfig["SurfTangentsY"] = SingleOptRes["SurfTangentsY"]
    InitConfig["SurfOrientations"] = SingleOptRes["SurfOrientations"]

    #Get Contact Locations
    TerminalConfig["Px"] = np.array(x_opt[var_idx_lv1["px"][0]:var_idx_lv1["px"][1]+1])
    TerminalConfig["Py"] = np.array(x_opt[var_idx_lv1["py"][0]:var_idx_lv1["py"][1]+1])
    TerminalConfig["Pz"] = np.array(x_opt[var_idx_lv1["pz"][0]:var_idx_lv1["pz"][1]+1])

    return InitConfig, TerminalConfig

#Computing Costs
#TotalCostType is a string: "|Acc|AM|AMdot|", can be any of them, default is "|Acc|AM|"
def compute_EH_Cost(Nk_local = 7, SingleOptResult = None, TotalCostType = "|Acc|AM|", robotmass = 95.0):
    #Constants
    G = 9.80665 #kg/m^2

    #initialise cost contrainer
    total_cost = 0;    cost_acc = 0;    cost_AM = 0;    cost_AMdot = 0

    #Get Varialbles
    #   Get Variable Index
    var_idx_lv1 = SingleOptResult["var_idx"]["Level1_Var_Index"]
    #   Get Optimization Result
    x_opt = SingleOptResult["opt_res"]
    #   Get Variables
    #   Switching Time
    Ts_res = np.array(x_opt[var_idx_lv1["Ts"][0]:var_idx_lv1["Ts"][1]+1])
    #   CoM
    x_res = np.array(x_opt[var_idx_lv1["x"][0]:var_idx_lv1["x"][1]+1]);    y_res = np.array(x_opt[var_idx_lv1["y"][0]:var_idx_lv1["y"][1]+1])
    z_res = np.array(x_opt[var_idx_lv1["z"][0]:var_idx_lv1["z"][1]+1])
    #   Angular Momentum
    Lx_res = np.array(x_opt[var_idx_lv1["Lx"][0]:var_idx_lv1["Lx"][1]+1]); Ly_res = np.array(x_opt[var_idx_lv1["Ly"][0]:var_idx_lv1["Ly"][1]+1])
    Lz_res = np.array(x_opt[var_idx_lv1["Lz"][0]:var_idx_lv1["Lz"][1]+1])
    #   Angular Momentum Rate
    Ldotx_res = np.array(x_opt[var_idx_lv1["Ldotx"][0]:var_idx_lv1["Ldotx"][1]+1]); Ldoty_res = np.array(x_opt[var_idx_lv1["Ldoty"][0]:var_idx_lv1["Ldoty"][1]+1])
    Ldotz_res = np.array(x_opt[var_idx_lv1["Ldotz"][0]:var_idx_lv1["Ldotz"][1]+1])
    #   Contact Force
    #   Left Foot Contact Point 1
    FL1x_res = np.array(x_opt[var_idx_lv1["FL1x"][0]:var_idx_lv1["FL1x"][1]+1]);   FL1y_res = np.array(x_opt[var_idx_lv1["FL1y"][0]:var_idx_lv1["FL1y"][1]+1])
    FL1z_res = np.array(x_opt[var_idx_lv1["FL1z"][0]:var_idx_lv1["FL1z"][1]+1])
    #   Left Foot Contact Point 2
    FL2x_res = np.array(x_opt[var_idx_lv1["FL2x"][0]:var_idx_lv1["FL2x"][1]+1]);   FL2y_res = np.array(x_opt[var_idx_lv1["FL2y"][0]:var_idx_lv1["FL2y"][1]+1])
    FL2z_res = np.array(x_opt[var_idx_lv1["FL2z"][0]:var_idx_lv1["FL2z"][1]+1])
    #   Left Foot Contact Point 3
    FL3x_res = np.array(x_opt[var_idx_lv1["FL3x"][0]:var_idx_lv1["FL3x"][1]+1]);   FL3y_res = np.array(x_opt[var_idx_lv1["FL3y"][0]:var_idx_lv1["FL3y"][1]+1])
    FL3z_res = np.array(x_opt[var_idx_lv1["FL3z"][0]:var_idx_lv1["FL3z"][1]+1])
    #   Left Foot Contact Point 4
    FL4x_res = np.array(x_opt[var_idx_lv1["FL4x"][0]:var_idx_lv1["FL4x"][1]+1]);   FL4y_res = np.array(x_opt[var_idx_lv1["FL4y"][0]:var_idx_lv1["FL4y"][1]+1])
    FL4z_res = np.array(x_opt[var_idx_lv1["FL4z"][0]:var_idx_lv1["FL4z"][1]+1])

    #   Right Foot Contact Point 1
    FR1x_res = np.array(x_opt[var_idx_lv1["FR1x"][0]:var_idx_lv1["FR1x"][1]+1]);   FR1y_res = np.array(x_opt[var_idx_lv1["FR1y"][0]:var_idx_lv1["FR1y"][1]+1])
    FR1z_res = np.array(x_opt[var_idx_lv1["FR1z"][0]:var_idx_lv1["FR1z"][1]+1])
    #   Right Foot Contact Point 2
    FR2x_res = np.array(x_opt[var_idx_lv1["FR2x"][0]:var_idx_lv1["FR2x"][1]+1]);   FR2y_res = np.array(x_opt[var_idx_lv1["FR2y"][0]:var_idx_lv1["FR2y"][1]+1])
    FR2z_res = np.array(x_opt[var_idx_lv1["FR2z"][0]:var_idx_lv1["FR2z"][1]+1])
    #   Right Foot Contact Point 3
    FR3x_res = np.array(x_opt[var_idx_lv1["FR3x"][0]:var_idx_lv1["FR3x"][1]+1]);   FR3y_res = np.array(x_opt[var_idx_lv1["FR3y"][0]:var_idx_lv1["FR3y"][1]+1])
    FR3z_res = np.array(x_opt[var_idx_lv1["FR3z"][0]:var_idx_lv1["FR3z"][1]+1])
    #   Right Foot Contact Point 4
    FR4x_res = np.array(x_opt[var_idx_lv1["FR4x"][0]:var_idx_lv1["FR4x"][1]+1]);   FR4y_res = np.array(x_opt[var_idx_lv1["FR4y"][0]:var_idx_lv1["FR4y"][1]+1])
    FR4z_res = np.array(x_opt[var_idx_lv1["FR4z"][0]:var_idx_lv1["FR4z"][1]+1])

    return None

#NOTE: Useless Now
#Add noise to the Initial State
#MaxNoise Level in mm
def add_Noise_to_Initial_State(Init_State = None, MaxNoiseLevel = 0.0):
    
    #NoisyInitState = copy.deepcopy(Init_State)

    #For CoM x
    noiseLevel = np.random.uniform(-MaxNoiseLevel,MaxNoiseLevel)/1000.0 #Convert to meters
    Init_State["x_init"] = Init_State["x_init"] + noiseLevel
    if Init_State["x_init"] < 0.0: #Shift two zero other infeasible problem
        Init_State["x_init"] = 0.0

    #For CoM y
    noiseLevel = np.random.uniform(-MaxNoiseLevel,MaxNoiseLevel)/1000.0 #Convert to meters
    Init_State["y_init"] = Init_State["y_init"] + noiseLevel

    #For CoM z
    noiseLevel = np.random.uniform(-MaxNoiseLevel,MaxNoiseLevel)/1000.0 #Convert to meters
    Init_State["z_init"] = Init_State["z_init"] + noiseLevel

    #For CoM xdot
    noiseLevel = np.random.uniform(-MaxNoiseLevel,MaxNoiseLevel)/1000.0 #Convert to meters
    Init_State["xdot_init"] = Init_State["xdot_init"] + noiseLevel
    if Init_State["xdot_init"] < 0.0: #Shift two zero other infeasible problem
        Init_State["xdot_init"] = 0.0

    #For CoM ydot
    noiseLevel = np.random.uniform(-MaxNoiseLevel,MaxNoiseLevel)/1000.0 #Convert to meters
    Init_State["ydot_init"] = Init_State["ydot_init"] + noiseLevel

    #For CoM zdot
    noiseLevel = np.random.uniform(-MaxNoiseLevel,MaxNoiseLevel)/1000.0 #Convert to meters
    Init_State["zdot_init"] = Init_State["zdot_init"] + noiseLevel

    #For Lx
    noiseLevel = np.random.uniform(-MaxNoiseLevel,MaxNoiseLevel)/1000.0 #Convert to meters
    Init_State["Lx_init"] = Init_State["Lx_init"] + noiseLevel

    #For Ly
    noiseLevel = np.random.uniform(-MaxNoiseLevel,MaxNoiseLevel)/1000.0 #Convert to meters
    Init_State["Ly_init"] = Init_State["Ly_init"] + noiseLevel

    #For Lz
    noiseLevel = np.random.uniform(-MaxNoiseLevel,MaxNoiseLevel)/1000.0 #Convert to meters
    Init_State["Lz_init"] = Init_State["Lz_init"] + noiseLevel

    #For Left InitContact
    #   Sample Noise Level
    noiseLevel = np.random.uniform(-MaxNoiseLevel,MaxNoiseLevel)/1000.0 #Convert to meters
    #   Compute local contact patch x vector and y vector
    #       x and y vector for the terrain in world frame (i.e. original world frame, shifted world frame)
    terrain_x_vec = (np.array([Init_State["LeftInitSurf"][0]]).T - np.array([Init_State["LeftInitSurf"][1]]).T)
    terrain_x_vec_normalised = terrain_x_vec/np.linalg.norm(terrain_x_vec)
    
    terrain_y_vec = np.array([Init_State["LeftInitSurf"][1]]).T - np.array([Init_State["LeftInitSurf"][2]]).T
    terrain_y_vec_normalised = terrain_y_vec/np.linalg.norm(terrain_y_vec)

    #       Get shifted Contact Location
    P_vec = np.array([[Init_State["PLx_init"],Init_State["PLy_init"],Init_State["PLz_init"]]]).T
    P_vec_noisy = P_vec + noiseLevel*terrain_x_vec_normalised + noiseLevel*terrain_y_vec_normalised
    #       Check if break kine constraint
    #           Get the origin (third point, index 2) of the Initial Contact Patch
    PatchOrigin = Init_State["LeftInitSurf"][2]
    #           Compute Four vertices of the foot
    P1 = P_vec_noisy + 0.11*np.array([Init_State["PL_init_TangentX"]]).T + 0.06*np.array([Init_State["PL_init_TangentY"]]).T
    P2 = P_vec_noisy + 0.11*np.array([Init_State["PL_init_TangentX"]]).T - 0.06*np.array([Init_State["PL_init_TangentY"]]).T
    P3 = P_vec_noisy - 0.11*np.array([Init_State["PL_init_TangentX"]]).T + 0.06*np.array([Init_State["PL_init_TangentY"]]).T
    P4 = P_vec_noisy - 0.11*np.array([Init_State["PL_init_TangentX"]]).T - 0.06*np.array([Init_State["PL_init_TangentY"]]).T
    #           Compare
    if P3[0][0] < PatchOrigin[0] or P3[1][0] < PatchOrigin[1]:
        P_vec_noisy = P_vec
    #       Assign Values
    Init_State["PLx_init"] = P_vec_noisy[0][0]
    Init_State["PLy_init"] = P_vec_noisy[1][0]
    Init_State["PLz_init"] = P_vec_noisy[2][0]

    #For Right InitContact
    #   Sample Noise Level
    noiseLevel = np.random.uniform(-MaxNoiseLevel,MaxNoiseLevel)/1000.0 #Convert to meters
    #   Compute local contact patch x vector and y vector
    #       x and y vector for the terrain in world frame (i.e. original world frame, shifted world frame)
    terrain_x_vec = (np.array([Init_State["RightInitSurf"][0]]).T - np.array([Init_State["RightInitSurf"][1]]).T)
    terrain_x_vec_normalised = terrain_x_vec/np.linalg.norm(terrain_x_vec)
    
    terrain_y_vec = np.array([Init_State["RightInitSurf"][1]]).T - np.array([Init_State["RightInitSurf"][2]]).T
    terrain_y_vec_normalised = terrain_y_vec/np.linalg.norm(terrain_y_vec)
    #       Get shifted Contact Location
    P_vec = np.array([[Init_State["PRx_init"],Init_State["PRy_init"],Init_State["PRz_init"]]]).T
    P_vec_noisy = P_vec + noiseLevel*terrain_x_vec_normalised + noiseLevel*terrain_y_vec_normalised
    #       Check if break kine constraint
    #           Get the origin (third point, index 2) of the Initial Contact Patch
    PatchOrigin = Init_State["RightInitSurf"][2]
    #           Compute Four vertices of the foot
    P1 = P_vec_noisy + 0.11*np.array([Init_State["PR_init_TangentX"]]).T + 0.06*np.array([Init_State["PR_init_TangentY"]]).T
    P2 = P_vec_noisy + 0.11*np.array([Init_State["PR_init_TangentX"]]).T - 0.06*np.array([Init_State["PR_init_TangentY"]]).T
    P3 = P_vec_noisy - 0.11*np.array([Init_State["PR_init_TangentX"]]).T + 0.06*np.array([Init_State["PR_init_TangentY"]]).T
    P4 = P_vec_noisy - 0.11*np.array([Init_State["PR_init_TangentX"]]).T - 0.06*np.array([Init_State["PR_init_TangentY"]]).T
    #           Compare
    if P3[0][0] < PatchOrigin[0] or P3[1][0] < PatchOrigin[1]:
        P_vec_noisy = P_vec
    #       Assign Values
    Init_State["PRx_init"] = P_vec_noisy[0][0]
    Init_State["PRy_init"] = P_vec_noisy[1][0]
    Init_State["PRz_init"] = P_vec_noisy[2][0]

    return Init_State