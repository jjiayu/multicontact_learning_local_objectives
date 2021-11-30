import numpy as np
from multicontact_learning_local_objectives.python.utils import *

#Collect from single Optimization Result
#   Shift_World_Frame = None: No Shift
#   Shift_World_Frame = InitCoM: Shift World frame to the CoM
#   Shift_World_Frame = InitSurfBorder: shift world frame to the left most border of the init patches in each optimization result
#   ContactRepresentationType = 1) 3DPoints 2) ConvexCombination
def getDataPoints(SingleOptRes = None, Shift_World_Frame = None, ContactRepresentationType = None, VectorScaleFactor = 1):

    #Get Init State 
    InitConfig, TerminalConfig = getInitTerminalConfig(SingleOptRes = SingleOptRes, Shift_World_Frame = Shift_World_Frame)

    #------------------------
    #Collect x (Input Vector) based on Shift World Frame Mode (if Stance Foot, the origin of the reference frame goes to the stance contact which will be all zeros)
    if Shift_World_Frame == None: #No transformation to local frames, then full input vector
        x = np.concatenate((InitConfig["x_init"],      InitConfig["y_init"],      InitConfig["z_init"],\
                            InitConfig["xdot_init"],   InitConfig["ydot_init"],   InitConfig["zdot_init"],\
                            InitConfig["Lx_init"],     InitConfig["Ly_init"],     InitConfig["Lz_init"], \
                            InitConfig["PLx_init"],    InitConfig["PLy_init"],    InitConfig["PLz_init"],\
                            InitConfig["PRx_init"],    InitConfig["PRy_init"],    InitConfig["PRz_init"],\
                            InitConfig["LeftSwingFlag"],\
                            InitConfig["LeftInitSurf"], InitConfig["RightInitSurf"], \
                            InitConfig["ContactSurfs"]), axis = None)
    elif Shift_World_Frame == "StanceFoot":
        #Decide Contact Location before Swing
        if (InitConfig["LeftSwingFlag"] == 1) and (InitConfig["RightSwingFlag"] == 0): #Swing Left Foot
            contact_before_swing_X = InitConfig["PLx_init"];     contact_before_swing_Y = InitConfig["PLy_init"]
            contact_before_swing_Z = InitConfig["PLz_init"]
        elif (InitConfig["LeftSwingFlag"] == 0) and (InitConfig["RightSwingFlag"] == 1): #Swing Right Foot
            contact_before_swing_X = InitConfig["PRx_init"];     contact_before_swing_Y = InitConfig["PRy_init"]
            contact_before_swing_Z = InitConfig["PRz_init"]
        else: 
            raise Exception("Unknow Leg Swing Indicators")
        #Collect Input Vector
        x = np.concatenate((InitConfig["x_init"],         InitConfig["y_init"],      InitConfig["z_init"],\
                            InitConfig["xdot_init"],     InitConfig["ydot_init"],   InitConfig["zdot_init"],\
                            InitConfig["Lx_init"],       InitConfig["Ly_init"],     InitConfig["Lz_init"],\
                            contact_before_swing_X,      contact_before_swing_Y,    contact_before_swing_Z,\
                            InitConfig["LeftSwingFlag"],\
                            InitConfig["LeftInitSurf"],  InitConfig["RightInitSurf"],\
                            InitConfig["ContactSurfs"]), axis = None)
    else:
        raise Exception("Unknown Mode of Shifting to Local Frame")

    #Collect y (Build Output Vector)

    if ContactRepresentationType == None:
        raise Exception("Unknown Contact Location Representation Type (ContactRepresentationType)")
    elif ContactRepresentationType == "3DPoints":
        y = np.concatenate((TerminalConfig["x_end"],       TerminalConfig["y_end"],       TerminalConfig["z_end"], \
                            TerminalConfig["xdot_end"],    TerminalConfig["ydot_end"],    TerminalConfig["zdot_end"],\
                            TerminalConfig["Lx_end"],      TerminalConfig["Ly_end"],      TerminalConfig["Lz_end"],\
                            TerminalConfig["Px"],     TerminalConfig["Py"],     TerminalConfig["Pz"]), axis = None)
    elif ContactRepresentationType == "ConvexCombination":
        #Generate Contact Location List
        P_land_ref = np.concatenate((TerminalConfig["Px"],TerminalConfig["Py"],TerminalConfig["Pz"]), axis = None)
        #Get the first Patch, and place vertex in a column vector fashion [v1,v2,v3,v4]
        FirstPatch = InitConfig["ContactSurfs"][0]
        #Compute Coefficient for convex combination
        coefs = Point3D_to_ConvexCombination(ContactLocation = P_land_ref, ContactSurf = FirstPatch)
        #Build y vector
        y = np.concatenate((TerminalConfig["x_end"],       TerminalConfig["y_end"],       TerminalConfig["z_end"], \
                            TerminalConfig["xdot_end"],    TerminalConfig["ydot_end"],    TerminalConfig["zdot_end"],\
                            TerminalConfig["Lx_end"],      TerminalConfig["Ly_end"],      TerminalConfig["Lz_end"],\
                            coefs), axis = None)
    elif ContactRepresentationType == "FollowRectangelBorder":
        P_land_ref = np.array([TerminalConfig["Px"],TerminalConfig["Py"],TerminalConfig["Pz"]]) #A vertical vector
        FirstPatch = InitConfig["ContactSurfs"][0] #Contact Surface Placed Row by Row
        FirstPatchOrientation = InitConfig["SurfOrientations"][0] #3x3 symmetric matrix
        FirstPatchLocalOrigin = np.array([FirstPatch[2]]) #The third point should be the local origin (index 2); NOTE: A row vector now, but in two dimension
        #Make Homogeneous Transformation
        HomoTran = np.hstack((FirstPatchOrientation,FirstPatchLocalOrigin.T))
        HomoTran = np.vstack((HomoTran,np.array([[0.0,0.0,0.0,1.0]])))
        #Transform Quantities in Local Frame (all quantities should have 0 z-axis, as all stay in the same plane and the origin should be 0,0,0)
        #   Contact Location (augumented with 1 for homotran)
        P_land_local_aug = np.linalg.inv(HomoTran)@np.vstack((P_land_ref,np.array([[1]])))
        #   Vertex 0-3 (augumented with 1 for homotran)
        Vertex0_local_aug = np.linalg.inv(HomoTran)@np.vstack((np.array([FirstPatch[0]]).T,np.array([[1]])))
        Vertex1_local_aug = np.linalg.inv(HomoTran)@np.vstack((np.array([FirstPatch[1]]).T,np.array([[1]])))
        Vertex2_local_aug = np.linalg.inv(HomoTran)@np.vstack((np.array([FirstPatch[2]]).T,np.array([[1]])))
        Vertex3_local_aug = np.linalg.inv(HomoTran)@np.vstack((np.array([FirstPatch[3]]).T,np.array([[1]])))
        #Get 2D Vectors
        P_land_local_2d = P_land_local_aug[0:2]
        Vertex0_local_2d = Vertex0_local_aug[0:2]
        Vertex1_local_2d = Vertex1_local_aug[0:2]
        Vertex2_local_2d = Vertex2_local_aug[0:2]
        Vertex3_local_2d = Vertex3_local_aug[0:2]
        #Get local axis
        #   x axis: First Vertex (index 0)- Second Vertex (index 1)
        local_x_vec = Vertex0_local_2d - Vertex1_local_2d
        #   y axis: Third Vertex (index 1) - Second Vertex (index 2)
        local_y_vec = Vertex1_local_2d - Vertex2_local_2d
        #Build Matrix and Compute Moving Extent
        TransMatrix = np.hstack((local_x_vec, local_y_vec))
        coefs = np.linalg.inv(TransMatrix)@P_land_local_2d
        #Build y vector
        y = np.concatenate((TerminalConfig["x_end"],       TerminalConfig["y_end"],       TerminalConfig["z_end"], \
                            TerminalConfig["xdot_end"],    TerminalConfig["ydot_end"],    TerminalConfig["zdot_end"],\
                            TerminalConfig["Lx_end"],      TerminalConfig["Ly_end"],      TerminalConfig["Lz_end"],\
                            coefs), axis = None)

    #-------------------------------
    #Add Contact Timing
    #Get Phase Duration
    var_Idx_lv1 = SingleOptRes["var_idx"]["Level1_Var_Index"]
    opt_res = SingleOptRes["opt_res"]

    Ts_vec = opt_res[var_Idx_lv1["Ts"][0]:var_Idx_lv1["Ts"][1]+1]

    InitDS_Dur = Ts_vec[-3]
    SS_Dur = Ts_vec[-2] - Ts_vec[-3]
    DS_Dur = Ts_vec[-1] - Ts_vec[-2]

    #Add to y vector
    y = np.concatenate((y, InitDS_Dur, SS_Dur, DS_Dur), axis = None)

    #--------------------------
    #Scale Vectors
    x = x*VectorScaleFactor;     y = y*VectorScaleFactor

    return x, y