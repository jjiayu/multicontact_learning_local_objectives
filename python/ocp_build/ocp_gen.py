#Module for generating OCP formulations

#NOTE: If reuse Ponton Code: we need to transofrm the kinematics constraints

import numpy as np #Numpy
import casadi as ca #Casadi
from .constraints import * #Constraints
import pickle
import os
import copy

#For computing Kinematics Polytopes
from sl1m.problem_definition import *
from sl1m.planner_scenarios.talos.constraints_shift import *

# def test_kine_constraint():
#     K_CoM_Right,k_CoM_Right = right_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
#     K_CoM_Left,k_CoM_Left = left_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))

#     #kinematicConstraints = genKinematicConstraints(left_foot_constraints,right_foot_constraints)
#     #K_CoM_Left_2 = kinematicConstraints[0][0]
#     #k_CoM_Left_2 = kinematicConstraints[0][1]
#     #K_CoM_Right_2 = kinematicConstraints[1][0]
#     #k_CoM_Right_2 = kinematicConstraints[1][1]

#     #print(K_CoM_Left - K_CoM_Left_2)
#     #print(K_CoM_Right - K_CoM_Right_2)
#     #print(k_CoM_Right - k_CoM_Right_2)
#     #print(k_CoM_Left - k_CoM_Left_2)

#     return None

#Function to build the first level
def NLP_SingleStep(m = 100.0, Nk_Local= 7, AngularDynamics = True, ParameterList = None, PhaseDuration_Limits = None, miu = 0.3, LocalObjMode = False):
    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constant Parameters
    #Parameters 
    G = 9.80665 #kg/m^2
    #Friction Coefficient
    #Force Limits
    F_bound = 400.0
    Fxlb = -F_bound; Fxub = F_bound
    Fylb = -F_bound; Fyub = F_bound
    Fzlb = -F_bound; Fzub = F_bound
    #Angular Momentum Bounds
    L_bound = 5; Llb = -L_bound; Lub = L_bound
    Ldot_bound = 5; Ldotlb = -Ldot_bound; Ldotub = Ldot_bound
    #CoM z heights limit in the world frame
    z_lowest = -5.0
    z_highest = 5.0
    #CoM Height with respect to Footstep Location (in the local stance frame, think about standstill pose)
    CoM_z_to_Foot_min = 0.6 #0.65
    CoM_z_to_Foot_max = 0.8 #0.75
    ##   Terrain Model
    ##       Flat Terrain
    #TerrainNorm = [0,0,1] 
    #TerrainTangentX = [1,0,0]
    #TerrainTangentY = [0,1,0]

    #-----------------------------------------------------------------------------------------------------------------------
    #Decide Motion Parameters
    #   Gait Pattern, Each action is followed up by a double support phase
    GaitPattern = ['InitialDouble','Swing','DoubleSupport'] #,'RightSupport','DoubleSupport','LeftSupport','DoubleSupport'
    #   Number of Phases
    Nphase = len(GaitPattern)
    #   Number of Steps
    Nstep = 1
    #   Compute Number of Total knots/ticks, but the enumeration start from 0 to N_K-1
    N_K = Nk_Local*Nphase + 1 #+1 the last knots to finalize the plan

    # #-----------------------------------------------------------------------------------------------------------------------
    # #Load kinematics Polytope
    # #   Not local than server
    # kinefilepath = "/home/jiayu/Desktop/multicontact_learning_local_objectives/python/kinematics_polytope/kinematics_constraints.p"
    # if os.path.isfile(kinefilepath) == False:
    #     kinefilepath = "/afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/kinematics_polytope/kinematics_constraints.p"

    # with open(kinefilepath, 'rb') as f:
    #     kinematics_constraints= pickle.load(f)
    
    # #CoM Polytopes
    # K_CoM_Right = kinematics_constraints["K_CoM_in_Right_Contact"];     k_CoM_Right = kinematics_constraints["k_CoM_in_Right_Contact"]
    # K_CoM_Left  = kinematics_constraints["K_CoM_in_Left_Contact"];      k_CoM_Left  = kinematics_constraints["k_CoM_in_Left_Contact"]

    # #Relative Footstep constraints
    # Q_rf_in_lf = kinematics_constraints["Q_Right_Contact_in_Left_Contact"];    q_rf_in_lf = kinematics_constraints["q_Right_Contact_in_Left_Contact"]
    # Q_lf_in_rf = kinematics_constraints["Q_Left_Contact_in_Right_Contact"];    q_lf_in_rf = kinematics_constraints["q_Left_Contact_in_Right_Contact"]

    #Get Kinematics Constraint for Talos
    #CoM kinematics constraint, give homogenous transformaiton (the last column seems like dont make a diff)
    K_CoM_Right,k_CoM_Right = right_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    K_CoM_Left,k_CoM_Left = left_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    #Another way
    #kinematicConstraints = genKinematicConstraints(left_foot_constraints,right_foot_constraints)
    #K_CoM_Left = kinematicConstraints[0][0];   k_CoM_Left = kinematicConstraints[0][1]
    #K_CoM_Right = kinematicConstraints[1][0];  k_CoM_Right = kinematicConstraints[1][1]
    
    #Relative Foot Constraint matrices
    #Relative foot constraint, give homogenous transformation (the last column seems like dont make a diff)
    Q_rf_in_lf,q_rf_in_lf = right_foot_in_lf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    Q_lf_in_rf,q_lf_in_rf = left_foot_in_rf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    #Another way
    #relativeConstraints = genFootRelativeConstraints(right_foot_in_lf_frame_constraints,left_foot_in_rf_frame_constraints)
    #Q_rf_in_lf = relativeConstraints[0][0];   q_rf_in_lf = relativeConstraints[0][1] #named lf in rf, but representing rf in lf
    #Q_lf_in_rf = relativeConstraints[1][0];   q_lf_in_rf = relativeConstraints[1][1] #named rf in lf, but representing lf in rf
    #-----------------------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Casadi Parameters
    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ParameterList["LeftSwingFlag"]
    ParaRightSwingFlag = ParameterList["RightSwingFlag"]
    #Initial Robot State
    #  Initial CoM State
    x_init = ParameterList["x_init"];       y_init = ParameterList["y_init"];       z_init = ParameterList["z_init"]
    #  Initial CoM Velocity
    xdot_init = ParameterList["xdot_init"]; ydot_init = ParameterList["ydot_init"]; zdot_init = ParameterList["zdot_init"]
    #  Initial Angular Momentum
    Lx_init = ParameterList["Lx_init"];     Ly_init = ParameterList["Ly_init"];     Lz_init = ParameterList["Lz_init"]
    #  Initial Left Foot Contact Location
    PLx_init = ParameterList["PLx_init"];   PLy_init = ParameterList["PLy_init"];   PLz_init = ParameterList["PLz_init"]
    PL_init = ca.vertcat(PLx_init,PLy_init,PLz_init)
    #  Initial Right Foot Contact Location
    PRx_init = ParameterList["PRx_init"];   PRy_init = ParameterList["PRy_init"];   PRz_init = ParameterList["PRz_init"]
    PR_init = ca.vertcat(PRx_init,PRy_init,PRz_init)

    ##Local Obj State (Get Them first, reach them by cost if we only have single step, non-use/random values if we have multiple steps lookahead)
    ##  Local Obj CoM State
    #x_obj = ParameterList["x_obj"];       y_obj = ParameterList["y_obj"];       z_obj = ParameterList["z_obj"]
    ##  Local Obj CoM Velocity
    #xdot_obj = ParameterList["xdot_obj"]; ydot_obj = ParameterList["ydot_obj"]; zdot_obj = ParameterList["zdot_obj"]
    ##  Local Obj Angular Momentum
    #Lx_obj = ParameterList["Lx_obj"];     Ly_obj = ParameterList["Ly_obj"];     Lz_obj = ParameterList["Lz_obj"]
    ##  Target Contact Location
    #Px_obj = ParameterList["Px_obj"];     Py_obj = ParameterList["Py_obj"];     Pz_obj = ParameterList["Pz_obj"]

    #Surfaces Parameters (Only the First One)
    SurfParas = ParameterList["SurfParas"]
    FirstSurfPara = SurfParas[0:19+1]
    #print(FirstSurPara)
    #   Process the Parameters
    #   FirstSurfK, the matrix - Confirmed Correctness
    FirstSurfK = FirstSurfPara[0:11+1]
    FirstSurfK = ca.reshape(FirstSurfK,3,4)
    FirstSurfK = FirstSurfK.T #NOTE: This process is due to casadi naming convention to have first row to be x1,x2,x3
    #   FirstSurfE, the vector for equality constraint
    FirstSurfE = FirstSurfPara[11+1:11+3+1]
    #   FirstSurfk, the vector fo inquality constraint
    FirstSurfk = FirstSurfPara[14+1:14+4+1]
    #   FirstSurfe, the vector fo inquality constraint
    FirstSurfe = FirstSurfPara[-1]

    #Terrain Tangents and Norms
    #Initial Contact Norm and Tangents
    PL_init_Norm = ParameterList["PL_init_Norm"];   PL_init_TangentX = ParameterList["PL_init_TangentX"];   PL_init_TangentY = ParameterList["PL_init_TangentY"]
    PR_init_Norm = ParameterList["PR_init_Norm"];   PR_init_TangentX = ParameterList["PR_init_TangentX"];   PR_init_TangentY = ParameterList["PR_init_TangentY"]
    
    #Future Contact Norm and Tangents
    SurfNorms = ParameterList["SurfNorms"]
    SurfTangentsX = ParameterList["SurfTangentsX"]
    SurfTangentsY = ParameterList["SurfTangentsY"]

    #Surface Orientations
    #Initial Contact Surface Orientations
    PL_init_Orientation = ParameterList["PL_init_Orientation"];  PL_init_Orientation = ca.reshape(PL_init_Orientation,3,3).T
    PR_init_Orientation = ParameterList["PR_init_Orientation"];  PR_init_Orientation = ca.reshape(PR_init_Orientation,3,3).T
    #
    SurfOriens = ParameterList["SurfOrientations"]
    FirstSurfOrientation = SurfOriens[0:9]
    FirstSurfOrientation = ca.reshape(FirstSurfOrientation,3,3).T

    #Timing Vectors
    InitDS_Ts_obj = ParameterList["InitDS_Ts_obj"]
    SS_Ts_obj = ParameterList["SS_Ts_obj"]
    DS_Ts_obj = ParameterList["DS_Ts_obj"]

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Variables and Bounds, Parameters
    #   CoM Position x, y, z
    x = ca.SX.sym('x',N_K);   x_lb = np.array([[0.0]*(x.shape[0]*x.shape[1])]);             x_ub = np.array([[50.0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    y = ca.SX.sym('y',N_K);   y_lb = np.array([[-1.0]*(y.shape[0]*y.shape[1])]);            y_ub = np.array([[1.0]*(y.shape[0]*y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    z = ca.SX.sym('z',N_K);   z_lb = np.array([[[z_lowest]]*(z.shape[0]*z.shape[1])]);    z_ub = np.array([[z_highest]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #   CoM Velocity x, y, z (Old value all -1.5 to 1.5)
    xdot = ca.SX.sym('xdot',N_K);   xdot_lb = np.array([[0.0]*(xdot.shape[0]*xdot.shape[1])]);   xdot_ub = np.array([[1.5]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    ydot = ca.SX.sym('ydot',N_K);   ydot_lb = np.array([[-1.5]*(ydot.shape[0]*ydot.shape[1])]);   ydot_ub = np.array([[1.5]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    zdot = ca.SX.sym('zdot',N_K);   zdot_lb = np.array([[-1.5]*(zdot.shape[0]*zdot.shape[1])]);   zdot_ub = np.array([[1.5]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #   Angular Momentum x, y, z
    Lx = ca.SX.sym('Lx',N_K);    Lx_lb = np.array([[Llb]*(Lx.shape[0]*Lx.shape[1])]);   Lx_ub = np.array([[Lub]*(Lx.shape[0]*Lx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ly = ca.SX.sym('Ly',N_K);    Ly_lb = np.array([[Llb]*(Ly.shape[0]*Ly.shape[1])]);   Ly_ub = np.array([[Lub]*(Ly.shape[0]*Ly.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Lz = ca.SX.sym('Lz',N_K);    Lz_lb = np.array([[Llb]*(Lz.shape[0]*Lz.shape[1])]);   Lz_ub = np.array([[Lub]*(Lz.shape[0]*Lz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #   Angular Momentum rate x, y, z
    Ldotx = ca.SX.sym('Ldotx',N_K);   Ldotx_lb = np.array([[Ldotlb]*(Ldotx.shape[0]*Ldotx.shape[1])]);   Ldotx_ub = np.array([[Ldotub]*(Ldotx.shape[0]*Ldotx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldoty = ca.SX.sym('Ldoty',N_K);   Ldoty_lb = np.array([[Ldotlb]*(Ldoty.shape[0]*Ldoty.shape[1])]);   Ldoty_ub = np.array([[Ldotub]*(Ldoty.shape[0]*Ldoty.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldotz = ca.SX.sym('Ldotz',N_K);   Ldotz_lb = np.array([[Ldotlb]*(Ldotz.shape[0]*Ldotz.shape[1])]);   Ldotz_ub = np.array([[Ldotub]*(Ldotz.shape[0]*Ldotz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Left Foot Forces
    #Left Foot Contact Point 1 x,y,z
    FL1x = ca.SX.sym('FL1x',N_K);   FL1x_lb = np.array([[Fxlb]*(FL1x.shape[0]*FL1x.shape[1])]);   FL1x_ub = np.array([[Fxub]*(FL1x.shape[0]*FL1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1y = ca.SX.sym('FL1y',N_K);   FL1y_lb = np.array([[Fylb]*(FL1y.shape[0]*FL1y.shape[1])]);   FL1y_ub = np.array([[Fyub]*(FL1y.shape[0]*FL1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1z = ca.SX.sym('FL1z',N_K);   FL1z_lb = np.array([[Fzlb]*(FL1z.shape[0]*FL1z.shape[1])]);   FL1z_ub = np.array([[Fzub]*(FL1z.shape[0]*FL1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Left Foot Contact Point 2 x,y,z
    FL2x = ca.SX.sym('FL2x',N_K);   FL2x_lb = np.array([[Fxlb]*(FL2x.shape[0]*FL2x.shape[1])]);   FL2x_ub = np.array([[Fxub]*(FL2x.shape[0]*FL2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2y = ca.SX.sym('FL2y',N_K);   FL2y_lb = np.array([[Fylb]*(FL2y.shape[0]*FL2y.shape[1])]);   FL2y_ub = np.array([[Fyub]*(FL2y.shape[0]*FL2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2z = ca.SX.sym('FL2z',N_K);   FL2z_lb = np.array([[Fzlb]*(FL2z.shape[0]*FL2z.shape[1])]);   FL2z_ub = np.array([[Fzub]*(FL2z.shape[0]*FL2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Left Foot Contact Point 3 x,y,z
    FL3x = ca.SX.sym('FL3x',N_K);   FL3x_lb = np.array([[Fxlb]*(FL3x.shape[0]*FL3x.shape[1])]);   FL3x_ub = np.array([[Fxub]*(FL3x.shape[0]*FL3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3y = ca.SX.sym('FL3y',N_K);   FL3y_lb = np.array([[Fylb]*(FL3y.shape[0]*FL3y.shape[1])]);   FL3y_ub = np.array([[Fyub]*(FL3y.shape[0]*FL3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3z = ca.SX.sym('FL3z',N_K);   FL3z_lb = np.array([[Fzlb]*(FL3z.shape[0]*FL3z.shape[1])]);   FL3z_ub = np.array([[Fzub]*(FL3z.shape[0]*FL3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Left Foot Contact Point 4 x,y,z
    FL4x = ca.SX.sym('FL4x',N_K);   FL4x_lb = np.array([[Fxlb]*(FL4x.shape[0]*FL4x.shape[1])]);   FL4x_ub = np.array([[Fxub]*(FL4x.shape[0]*FL4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4y = ca.SX.sym('FL4y',N_K);   FL4y_lb = np.array([[Fylb]*(FL4y.shape[0]*FL4y.shape[1])]);   FL4y_ub = np.array([[Fyub]*(FL4y.shape[0]*FL4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4z = ca.SX.sym('FL4z',N_K);   FL4z_lb = np.array([[Fzlb]*(FL4z.shape[0]*FL4z.shape[1])]);   FL4z_ub = np.array([[Fzub]*(FL4z.shape[0]*FL4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Right Foot Forces
    #Right Foot Contact Point 1 x,y,z
    FR1x = ca.SX.sym('FR1x',N_K);   FR1x_lb = np.array([[Fxlb]*(FR1x.shape[0]*FR1x.shape[1])]);   FR1x_ub = np.array([[Fxub]*(FR1x.shape[0]*FR1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1y = ca.SX.sym('FR1y',N_K);   FR1y_lb = np.array([[Fylb]*(FR1y.shape[0]*FR1y.shape[1])]);   FR1y_ub = np.array([[Fyub]*(FR1y.shape[0]*FR1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1z = ca.SX.sym('FR1z',N_K);   FR1z_lb = np.array([[Fzlb]*(FR1z.shape[0]*FR1z.shape[1])]);   FR1z_ub = np.array([[Fzub]*(FR1z.shape[0]*FR1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Right Foot Contact Point 2 x,y,z
    FR2x = ca.SX.sym('FR2x',N_K);   FR2x_lb = np.array([[Fxlb]*(FR2x.shape[0]*FR2x.shape[1])]);   FR2x_ub = np.array([[Fxub]*(FR2x.shape[0]*FR2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2y = ca.SX.sym('FR2y',N_K);   FR2y_lb = np.array([[Fylb]*(FR2y.shape[0]*FR2y.shape[1])]);   FR2y_ub = np.array([[Fyub]*(FR2y.shape[0]*FR2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2z = ca.SX.sym('FR2z',N_K);   FR2z_lb = np.array([[Fzlb]*(FR2z.shape[0]*FR2z.shape[1])]);   FR2z_ub = np.array([[Fzub]*(FR2z.shape[0]*FR2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Right Foot Contact Point 3 x,y,z
    FR3x = ca.SX.sym('FR3x',N_K);   FR3x_lb = np.array([[Fxlb]*(FR3x.shape[0]*FR3x.shape[1])]);   FR3x_ub = np.array([[Fxub]*(FR3x.shape[0]*FR3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3y = ca.SX.sym('FR3y',N_K);   FR3y_lb = np.array([[Fylb]*(FR3y.shape[0]*FR3y.shape[1])]);   FR3y_ub = np.array([[Fyub]*(FR3y.shape[0]*FR3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3z = ca.SX.sym('FR3z',N_K);   FR3z_lb = np.array([[Fzlb]*(FR3z.shape[0]*FR3z.shape[1])]);   FR3z_ub = np.array([[Fzub]*(FR3z.shape[0]*FR3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Right Foot Contact Point 4 x,y,z
    FR4x = ca.SX.sym('FR4x',N_K);   FR4x_lb = np.array([[Fxlb]*(FR4x.shape[0]*FR4x.shape[1])]);   FR4x_ub = np.array([[Fxub]*(FR4x.shape[0]*FR4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4y = ca.SX.sym('FR4y',N_K);   FR4y_lb = np.array([[Fylb]*(FR4y.shape[0]*FR4y.shape[1])]);   FR4y_ub = np.array([[Fyub]*(FR4y.shape[0]*FR4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4z = ca.SX.sym('FR4z',N_K);   FR4z_lb = np.array([[Fzlb]*(FR4z.shape[0]*FR4z.shape[1])]);   FR4z_ub = np.array([[Fzub]*(FR4z.shape[0]*FR4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements

    #   Contact Location Sequence
    px = [];   px_lb = [];   px_ub = []
    py = [];   py_lb = [];   py_ub = []
    pz = [];   pz_lb = [];   pz_ub = []
    #The first level only make the first step, which we enumerate as step 0
    for stepIdx in range(Nstep):
        pxtemp = ca.SX.sym('px'+str(stepIdx));  px.append(pxtemp);   px_lb.append(np.array([-1.0]));   px_ub.append(np.array([50.0]))
        pytemp = ca.SX.sym('py'+str(stepIdx));  py.append(pytemp);   py_lb.append(np.array([-2.0]));   py_ub.append(np.array([2.0]))
        pztemp = ca.SX.sym('pz'+str(stepIdx));  pz.append(pztemp);   pz_lb.append(np.array([-5.0]));   pz_ub.append(np.array([5.0]))

    #Switching Time Vector
    Ts = [];   Ts_lb = [];   Ts_ub = []
    for n_phase in range(Nphase):
        #Switching Time counts from t1, as t0 = 0
        Tstemp = ca.SX.sym('Ts'+str(n_phase+1));   Ts.append(Tstemp);   
        #Define Phase Duration Limits based on Phase Duration Limits
        Ts_lb.append(np.array([PhaseDuration_Limits["DoubleSupport_Min"]*0.1]));   
        Ts_ub.append(np.array([(PhaseDuration_Limits["DoubleSupport_Max"]+PhaseDuration_Limits["SingleSupport_Max"]+PhaseDuration_Limits["DoubleSupport_Max"])*1.5]))

    #   Collect all Decision Variables
    DecisionVars = ca.vertcat(x,    y,    z,    xdot,    ydot,    zdot,
                              Lx,   Ly,   Lz,   Ldotx,   Ldoty,   Ldotz,
                              FL1x, FL1y, FL1z, FL2x, FL2y, FL2z, FL3x, FL3y, FL3z, FL4x, FL4y, FL4z,
                              FR1x, FR1y, FR1z, FR2x, FR2y, FR2z, FR3x, FR3y, FR3z, FR4x, FR4y, FR4z,
                              *px,  *py,  *pz,
                              *Ts)
    DecisionVarsShape = DecisionVars.shape

    #   Collect all lower bound and upper bound
    #   Lower Bounds for Decision Variables
    DecisionVars_lb = (x_lb,    y_lb,    z_lb,    xdot_lb,    ydot_lb,    zdot_lb,
                       Lx_lb,   Ly_lb,   Lz_lb,   Ldotx_lb,   Ldoty_lb,   Ldotz_lb,
                       FL1x_lb, FL1y_lb, FL1z_lb, FL2x_lb, FL2y_lb, FL2z_lb, FL3x_lb, FL3y_lb, FL3z_lb, FL4x_lb, FL4y_lb, FL4z_lb,
                       FR1x_lb, FR1y_lb, FR1z_lb, FR2x_lb, FR2y_lb, FR2z_lb, FR3x_lb, FR3y_lb, FR3z_lb, FR4x_lb, FR4y_lb, FR4z_lb,
                       px_lb,   py_lb,   pz_lb,
                       Ts_lb)
    DecisionVars_lb = np.concatenate(DecisionVars_lb,axis=None)
    
    #   Upper Bounds for Decision Variables
    DecisionVars_ub = (x_ub,    y_ub,    z_ub,    xdot_ub,    ydot_ub,    zdot_ub,
                       Lx_ub,   Ly_ub,   Lz_ub,   Ldotx_ub,   Ldoty_ub,   Ldotz_ub,
                       FL1x_ub, FL1y_ub, FL1z_ub, FL2x_ub, FL2y_ub, FL2z_ub, FL3x_ub, FL3y_ub, FL3z_ub, FL4x_ub, FL4y_ub, FL4z_ub,
                       FR1x_ub, FR1y_ub, FR1z_ub, FR2x_ub, FR2y_ub, FR2z_ub, FR3x_ub, FR3y_ub, FR3z_ub, FR4x_ub, FR4y_ub, FR4z_ub,
                       px_ub,   py_ub,   pz_ub,
                       Ts_ub)
    DecisionVars_ub = np.concatenate(DecisionVars_ub,axis=None)


    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constrains and Running Cost

    #Constraint Container and Cost
    g = [];   glb = [];   gub = []
    J = 0.0

    #Time Span Setup for integration
    tau_upper_limit = 1.0
    tauStepLength = tau_upper_limit/(N_K-1) #Get the interval length, total number of knots - 1

    #Initial Conditions -> using standard equality constraints a = b
    #   Initial CoM x, y, z
    g, glb, gub = std_eq_constraint(a = ca.vertcat(x[0],   y[0],   z[0]), 
                                    b = ca.vertcat(x_init, y_init, z_init), g = g, glb= glb, gub = gub)
    #   Initial CoM Velocity x, y, z
    g, glb, gub = std_eq_constraint(a = ca.vertcat(xdot[0],   ydot[0],   zdot[0]), 
                                    b = ca.vertcat(xdot_init, ydot_init, zdot_init), g = g, glb= glb, gub = gub)
    #   Initial Angular Momentum x, y, z
    g, glb, gub = std_eq_constraint(a = ca.vertcat(Lx[0],   Ly[0],   Lz[0]), 
                                    b = ca.vertcat(Lx_init, Ly_init, Lz_init), g = g, glb= glb, gub = gub)

    ##Reaching Local Obj State
    #if OneStep == True:
    #    if ObjReachingType == "cost":
    #        weight = 1000
    #        J = J + weight*(x[-1]    - x_obj)**2    + weight*(y[-1]    - y_obj)**2    + weight*(z[-1]    - z_obj)**2    + \
    #                weight*(xdot[-1] - xdot_obj)**2 + weight*(ydot[-1] - ydot_obj)**2 + weight*(zdot[-1] - zdot_obj)**2 + \
    #                weight*(Lx[-1]   - Lx_obj)**2   + weight*(Ly[-1]   - Ly_obj)**2   + weight*(Lz[-1]   - Lz_obj)**2 + \
    #                weight*(px[0]    - Px_obj)**2   + weight*(py[0]    - Py_obj)**2   + weight*(pz[0]    - Pz_obj)**2
    #    elif ObjReachingType == "constraint":
    #        #   Obj CoM x, y, z
    #        g, glb, gub = std_eq_constraint(a = ca.vertcat(x[-1],  y[-1],  z[-1]),
    #                                        b = ca.vertcat(x_obj,  y_obj,  z_obj), g = g, glb= glb, gub = gub)
    #        #   Obj CoM Velocity x, y ,z
    #        g, glb, gub = std_eq_constraint(a = ca.vertcat(xdot[-1],  ydot[-1],  zdot[-1]), 
    #                                        b = ca.vertcat(xdot_obj,  ydot_obj,  zdot_obj), g = g, glb= glb, gub = gub)
    #        #   Obj Angular Momentum x, y, z
    #        g, glb, gub = std_eq_constraint(a = ca.vertcat(Lx[-1],  Ly[-1],  Lz[-1]), 
    #                                        b = ca.vertcat(Lx_obj,  Ly_obj,  Lz_obj), g = g, glb= glb, gub = gub)
    #        #   Contact Locations
    #        g, glb, gub = std_eq_constraint(a = ca.vertcat(px[0],   py[0],   pz[0]), 
    #                                        b = ca.vertcat(Px_obj,  Py_obj,  Pz_obj), g = g, glb= glb, gub = gub)            
    #    elif ObjReachingType == None:
    #        print("No cost for Reaching the Local Obj, Either try to reach the far goal with Single NLP, or multistep NLP")
    #    else:
    #        raise Exception("Undefined Local Obj Reaching Type (given OneStep = True); Valid Types are cost_localobj and constraint")

    #Constraints for all the knots
    #Loop over all Phases (Knots)
    for Nph in range(Nphase):
        #Decide Number of Knots
        if Nph == Nphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local       

        #Decide Time Vector
        if Nph == 0: #first phase
            h = tauStepLength*Nphase*(Ts[Nph]-0)
        else: #other phases
            h = tauStepLength*Nphase*(Ts[Nph]-Ts[Nph-1]) 

        for Local_k_Count in range(Nk_ThisPhase):
            #Get knot number across the entire time line
            k = Nph*Nk_Local + Local_k_Count
            #print(k)

            #------------------------------------------
            #Build useful vectors
            #   Forces of Left and Right foot at knot k
            FL1_k = ca.vertcat(FL1x[k],FL1y[k],FL1z[k]); FL2_k = ca.vertcat(FL2x[k],FL2y[k],FL2z[k]); FL3_k = ca.vertcat(FL3x[k],FL3y[k],FL3z[k]); FL4_k = ca.vertcat(FL4x[k],FL4y[k],FL4z[k])
            FR1_k = ca.vertcat(FR1x[k],FR1y[k],FR1z[k]); FR2_k = ca.vertcat(FR2x[k],FR2y[k],FR2z[k]); FR3_k = ca.vertcat(FR3x[k],FR3y[k],FR3z[k]); FR4_k = ca.vertcat(FR4x[k],FR4y[k],FR4z[k])
            #   CoM at knot k
            CoM_k = ca.vertcat(x[k],y[k],z[k])
            #   Angular Momentum
            Ldot_k = ca.vertcat(Ldotx[k],Ldoty[k],Ldotz[k])
            #-------------------------------------------

            #-------------------------------------------
            #Phase dependent Constraints (CoM Kinematics and Angular Dynamics)
            if GaitPattern[Nph]=='InitialDouble':
                #Kinematics Constraint
                #   CoM in the Left foot
                g, glb, gub = CoM_Kinematics(SwingLegIndicator = None, CoM_k = CoM_k, P = PL_init, K_polytope = K_CoM_Left,  k_polytope = k_CoM_Left,  
                                             ContactFrameOrientation = PL_init_Orientation,
                                             g = g, glb = glb, gub = gub)
                #   CoM in the Right foot
                g, glb, gub = CoM_Kinematics(SwingLegIndicator = None, CoM_k = CoM_k, P = PR_init, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                             ContactFrameOrientation = PR_init_Orientation,
                                             g = g, glb = glb, gub = gub)

                #   CoM Height Constraint (Left foot)
                g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = None, CoM_k = CoM_k, P = PL_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                       ContactFrameOrientation = PL_init_Orientation,
                                                       g = g, glb = glb, gub = gub)
                #   CoM Height Constraint (Right foot)
                g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = None, CoM_k = CoM_k, P = PR_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                       ContactFrameOrientation = PR_init_Orientation,
                                                       g = g, glb = glb, gub = gub)
                
                
                #Angular Dynamics
                #Definition of Contact Points of a foot
                #P3----------------P1
                #|                  |
                #|                  |
                #|                  |
                #P4----------------P2
                if AngularDynamics == True:
                    if k<N_K-1: #N_K-1 the enumeration of the last knot, k<N_K-1 the one before the last knot
                        g, glb, gub = Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator = None, 
                                                                          Ldot_k = Ldot_k, CoM_k = CoM_k,
                                                                          PL = PL_init, PL_TangentX = PL_init_TangentX, PL_TangentY = PL_init_TangentY, 
                                                                          PR = PR_init, PR_TangentX = PR_init_TangentX, PR_TangentY = PR_init_TangentY, 
                                                                          FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, 
                                                                          FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k, g = g, glb = glb, gub = gub)
                
                #Unilateral Force Constraints for all Suppport foot
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = None, F_k = FL1_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = None, F_k = FL2_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = None, F_k = FL3_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = None, F_k = FL4_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = None, F_k = FR1_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = None, F_k = FR2_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = None, F_k = FR3_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = None, F_k = FR4_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)

                #Friction Cone
                #Initial phase, no Leg Swing First Inidcators
                g, glb, gub = FrictionCone(SwingLegIndicator = None, F_k = FL1_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = None, F_k = FL2_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = None, F_k = FL3_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = None, F_k = FL4_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)

                g, glb, gub = FrictionCone(SwingLegIndicator = None, F_k = FR1_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = None, F_k = FR2_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = None, F_k = FR3_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = None, F_k = FR4_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)

            elif GaitPattern[Nph]=='Swing':
                #Kinematics Constraint and Angular Dynamics Constraint
                
                #----------------
                #Case1: IF LEFT Foot is SWING (RIGHT FOOT is STATIONARY)
                SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                #Kinematics Constraint
                #   CoM in the RIGHT Foot
                g, glb, gub, = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                              CoM_k = CoM_k, P = PR_init, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                              ContactFrameOrientation = PR_init_Orientation,
                                              g = g, glb = glb, gub = gub)
                
                #   CoM Height Constraint (Right Foot)
                g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PR_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                       ContactFrameOrientation = PR_init_Orientation,
                                                       g = g, glb = glb, gub = gub)

                # #Angular Dynamics (Right Support)
                if AngularDynamics == True:
                    if k<N_K-1:
                        g, glb, gub = Angular_Momentum_Rate_Swing(SwingLegIndicator = SwingLegFlag, 
                                                                  Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                  P = PR_init,  P_TangentX = PR_init_TangentX, P_TangentY = PR_init_TangentY, 
                                                                  F1_k = FR1_k, F2_k = FR2_k, F3_k = FR3_k, F4_k = FR4_k, g = g, glb = glb, gub = gub)
                
                #Unilateral Constraint
                # if the Left foot Swings, then the right foot should have unilateral constraints
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)

                #Zero Force Constrain
                # if the Left Foot Swings, then the Left foot should have zero forces
                g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, g = g, glb = glb, gub = gub)
                g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, g = g, glb = glb, gub = gub)
                g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, g = g, glb = glb, gub = gub)
                g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, g = g, glb = glb, gub = gub)

                #Friction Cone Constraint
                #If swing the Left foot first, then friction cone enforced on the Right Foot
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)

                #-----------------
                #Case 2: If RIGHT foot is SWING (LEFT is STATIONARY), Then LEFT Foot is the Support FOOT
                SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                #Kinematics Constraint
                #   CoM in the Left foot
                g, glb, gub, = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                              CoM_k = CoM_k, P = PL_init, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                              ContactFrameOrientation = PL_init_Orientation,
                                              g = g, glb = glb, gub = gub)
                #   CoM Height Constraint (Left Foot)
                g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PL_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                       ContactFrameOrientation = PL_init_Orientation,
                                                       g = g, glb = glb, gub = gub)

                # #Angular Dynamics (Left Support)
                if AngularDynamics == True:
                    if k<N_K-1:
                        g, glb, gub = Angular_Momentum_Rate_Swing(SwingLegIndicator = SwingLegFlag, 
                                                                  Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                  P = PL_init, P_TangentX = PL_init_TangentX, P_TangentY = PL_init_TangentY, 
                                                                  F1_k = FL1_k, F2_k = FL2_k, F3_k = FL3_k, F4_k = FL4_k, g = g, glb = glb, gub = gub)
                #Unilateral Constraint
                # if the Right foot Swings, then the Left foot should have unilateral constraints
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)

                #Zero Force Constrain
                # if the Right Foot Swing, then the Right foot should have zero forces
                g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, g = g, glb = glb, gub = gub)
                g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, g = g, glb = glb, gub = gub)
                g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, g = g, glb = glb, gub = gub)
                g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, g = g, glb = glb, gub = gub)

                #Friction Cone Constraint
                #If swing the Right Foot First, then the friction cone enforced on the Left Foot
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)

            elif GaitPattern[Nph]=='DoubleSupport':
                #Get Info for Landed Foot
                #   Landed Position
                P_land = ca.vertcat(*px,*py,*pz) #Moved/Swing Foot landing position
                #   Terrain Tangent and Norm of the Landed Foot
                P_land_Norm = SurfNorms[0:3]
                P_land_TangentX = SurfTangentsX[0:3]
                P_land_TangentY = SurfTangentsY[0:3]
                #----------------
                #Case 1
                #IF LEFT Foot is SWING (RIGHT FOOT - InitFoot is STATIONARY, Left foot becomes P_land)
                SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg swinged
                #Kinematics Constraint
                #   CoM in the LEFT foot Polyttope (Moved/Swing/landed - P_land)
                g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                             CoM_k = CoM_k, P = P_land, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                             ContactFrameOrientation = FirstSurfOrientation,
                                             g = g, glb = glb, gub = gub)
                #   CoM in the RIGHT Foot Polytope (Init Foot)
                g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                             CoM_k = CoM_k, P = PR_init, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                             ContactFrameOrientation = PR_init_Orientation,
                                             g = g, glb = glb, gub = gub)
                
                #   CoM Height Constraint (Left foot Moving\Landed Foot)
                g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = P_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                       ContactFrameOrientation = FirstSurfOrientation,
                                                       g = g, glb = glb, gub = gub)
                #   CoM Height Constraint (Right Foot - Init Foot)
                g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PR_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                       ContactFrameOrientation = PR_init_Orientation,
                                                       g = g, glb = glb, gub = gub)

                #Angular Dynamics (Double Support)
                if AngularDynamics == True:   
                    if k<N_K-1:
                        g, glb, gub = Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator = SwingLegFlag, 
                                                                          Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                          PL = P_land,  PL_TangentX = P_land_TangentX,  PL_TangentY = P_land_TangentY, 
                                                                          PR = PR_init, PR_TangentX = PR_init_TangentX, PR_TangentY = PR_init_TangentY, 
                                                                          FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, 
                                                                          FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k, g = g, glb = glb, gub = gub)
                
                #Unilater Constraints
                # if swing the Left foot first, then the Left foot obey unilateral constraint on the New SurfaceNorm
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = P_land_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = P_land_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = P_land_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = P_land_Norm, g = g, glb = glb, gub = gub)
                # Then the Right foot oby the unilateral constraints on the Init Surface Norm
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                
                #Friction Cone
                #If Swing the Left foot first, then the Left foot obey the friction cone constraint in the new landing place
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = P_land_TangentX, TerrainTangentY = P_land_TangentY, TerrainNorm = P_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = P_land_TangentX, TerrainTangentY = P_land_TangentY, TerrainNorm = P_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = P_land_TangentX, TerrainTangentY = P_land_TangentY, TerrainNorm = P_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = P_land_TangentX, TerrainTangentY = P_land_TangentY, TerrainNorm = P_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                #Then the Right foot obey the fricition cone constraint in the initial landing place
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)

                #-----------------
                #Case 2
                #if RIGHT Foot is SWING (LEFT FOOT - Init Foot is STATIONARY, Right foot becomese P_land)
                SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinged
                #Kinematics Constraint
                #   CoM in the Left foot (Init (Left) Foot)
                g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                             CoM_k = CoM_k, P = PL_init, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                             ContactFrameOrientation = PL_init_Orientation,
                                             g = g, glb = glb, gub = gub)
                #   CoM in the Right foot (Moved/Swing - PR_k) 
                g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                             CoM_k = CoM_k, P = P_land, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                             ContactFrameOrientation = FirstSurfOrientation,
                                             g = g, glb = glb, gub = gub)
                
                #   CoM Height Constraint (Init (Left) foot)
                g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PL_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                       ContactFrameOrientation = PL_init_Orientation,
                                                       g = g, glb = glb, gub = gub)
                #   CoM Height Constraint (Right foot Moving Foot/Land Foot)
                g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = P_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                       ContactFrameOrientation = FirstSurfOrientation,
                                                       g = g, glb = glb, gub = gub)

                if AngularDynamics == True:
                    if k<N_K-1:
                        g, glb, gub = Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator = SwingLegFlag, 
                                                                          Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                          PL = PL_init, PL_TangentX = PL_init_TangentX, PL_TangentY = PL_init_TangentY, 
                                                                          PR = P_land,  PR_TangentX = P_land_TangentX,  PR_TangentY = P_land_TangentY, 
                                                                          FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, 
                                                                          FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k, g = g, glb = glb, gub = gub)

                #Unilater Constraints
                # if swing the Right foot first,
                # Then the Left foot oby the unilateral constraints on the Init Surface Norm
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                # the Right foot obey unilateral constraint on the New SurfaceNorm
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = P_land_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = P_land_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = P_land_Norm, g = g, glb = glb, gub = gub)
                g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = P_land_Norm, g = g, glb = glb, gub = gub)
   

                #Friction Cone
                #If swing the Right foot first,
                # the Left foot obey the constaint on the initial landing place
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)

                # the Right foot obey the friction cone constraint in the new landing place
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = P_land_TangentX, TerrainTangentY = P_land_TangentY, TerrainNorm = P_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = P_land_TangentX, TerrainTangentY = P_land_TangentY, TerrainNorm = P_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = P_land_TangentX, TerrainTangentY = P_land_TangentY, TerrainNorm = P_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = P_land_TangentX, TerrainTangentY = P_land_TangentY, TerrainNorm = P_land_Norm, miu = miu, g = g, glb = glb, gub = gub)

            else:
                raise Exception("Unknow Phase Name")
            # #-------------------------------------
            # #Dynamics Constraint
            if k < N_K - 1: #N_K - 1 the enumeration of the last knot, -1 the knot before the last knot
                #First-order Dynamics CoM x, y, z
                g, glb, gub = First_Order_Integrator(next_state = x[k+1], cur_state = x[k], cur_derivative = xdot[k], h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = y[k+1], cur_state = y[k], cur_derivative = ydot[k], h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = z[k+1], cur_state = z[k], cur_derivative = zdot[k], h = h, g = g, glb = glb, gub = gub)
                #First-order Dynamics CoMdot x, y, z
                g, glb, gub = First_Order_Integrator(next_state = xdot[k+1], cur_state = xdot[k], 
                                                     cur_derivative = FL1x[k]/m + FL2x[k]/m + FL3x[k]/m + FL4x[k]/m + FR1x[k]/m + FR2x[k]/m + FR3x[k]/m + FR4x[k]/m, 
                                                     h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = ydot[k+1], cur_state = ydot[k], 
                                                     cur_derivative = FL1y[k]/m + FL2y[k]/m + FL3y[k]/m + FL4y[k]/m + FR1y[k]/m + FR2y[k]/m + FR3y[k]/m + FR4y[k]/m, 
                                                     h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = zdot[k+1], cur_state = zdot[k], 
                                                     cur_derivative = FL1z[k]/m + FL2z[k]/m + FL3z[k]/m + FL4z[k]/m + FR1z[k]/m + FR2z[k]/m + FR3z[k]/m + FR4z[k]/m - G, 
                                                     h = h, g = g, glb = glb, gub = gub)
                #First-Order Dynamics L x, y, z
                g, glb, gub = First_Order_Integrator(next_state = Lx[k+1], cur_state = Lx[k], cur_derivative = Ldotx[k], h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = Ly[k+1], cur_state = Ly[k], cur_derivative = Ldoty[k], h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = Lz[k+1], cur_state = Lz[k], cur_derivative = Ldotz[k], h = h, g = g, glb = glb, gub = gub)
            
            #Cost Terms
            if k < N_K - 1:
                #with Acceleration
                J = J + h*(FL1x[k]/m + FL2x[k]/m + FL3x[k]/m + FL4x[k]/m + FR1x[k]/m + FR2x[k]/m + FR3x[k]/m + FR4x[k]/m)**2 + \
                        h*(FL1y[k]/m + FL2y[k]/m + FL3y[k]/m + FL4y[k]/m + FR1y[k]/m + FR2y[k]/m + FR3y[k]/m + FR4y[k]/m)**2 + \
                        h*(FL1z[k]/m + FL2z[k]/m + FL3z[k]/m + FL4z[k]/m + FR1z[k]/m + FR2z[k]/m + FR3z[k]/m + FR4z[k]/m - G)**2
                #with Angular Momentum
                J = J + h*Lx[k]**2 + h*Ly[k]**2 + h*Lz[k]**2
                #With Angular momentum rate
                #J = J + h*Ldotx[k]**2 + h*Ldoty[k]**2 + h*Ldotz[k]**2

    #Relative Foot Constraints
    # #   Check for init footstep locations
    # #   Rf in Lf
    # g, glb, gub = Relative_Foot_Kinematics(SwingLegIndicator = None, 
    #                                        p_next = PR_init, p_cur = PL_init, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, ContactFrameOrientation = PL_init_Orientation,
    #                                        g = g, glb = glb, gub = gub)
    # #   Lf in Rf
    # g, glb, gub = Relative_Foot_Kinematics(SwingLegIndicator = None, 
    #                                        p_next = PL_init, p_cur = PR_init, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, ContactFrameOrientation = PR_init_Orientation,
    #                                        g = g, glb = glb, gub = gub)

    #   For the Double Support Phase (at landing), the landed swing foot position
    p_next = ca.vertcat(*px,*py,*pz)
    #Case 1--------
    #If LEFT foot is SWING (RIGHT is STATIONARY)
    SwingLegFlag = ParaLeftSwingFlag
    #Then LEFT Foot should Stay in the polytpe of the RIGHT FOOT
    g, glb, gub = Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                           p_next = p_next, p_cur = PR_init, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, ContactFrameOrientation = PR_init_Orientation,
                                           g = g, glb = glb, gub = gub)
    #Right - Stationary foot should also inthe polytope of the Swing foot
    g, glb, gub = Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                           p_next = PR_init, p_cur = p_next, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, ContactFrameOrientation = FirstSurfOrientation,
                                           g = g, glb = glb, gub = gub)
    #Case 2---------
    #If RIGHT foot is SWING (LEFT is STATIONARY)
    SwingLegFlag = ParaRightSwingFlag
    #Then RIGHT Foot should stay in the polytope of the LEFT Foot
    g, glb, gub = Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                           p_next = p_next, p_cur = PL_init, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, ContactFrameOrientation = PL_init_Orientation,
                                           g = g, glb = glb, gub = gub)
    #Left - Stationary foot should also in the polytope of the Swing foot
    g, glb, gub = Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                           p_next = PL_init, p_cur = p_next, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, ContactFrameOrientation = FirstSurfOrientation,
                                           g = g, glb = glb, gub = gub)


    #FootStep Location Constraint (On the Patch) -> Only One Step
    #P3----------------P1
    #|                  |
    #|                  |
    #|                  |
    #P4----------------P2
    
    # Norm at the new landing surface
    p_land = ca.vertcat(*px,*py,*pz)
    p_land_norm = SurfNorms[0:3]
    p_land_TangentX = SurfTangentsX[0:3]
    p_land_TangentY = SurfTangentsY[0:3]

    g, glb, gub = Stay_on_Surf(P = p_land, P_TangentX = p_land_TangentX, P_TangentY = p_land_TangentY, 
                               ineq_K = FirstSurfK, ineq_k = FirstSurfk, eq_E = FirstSurfE, eq_e = FirstSurfe, g = g, glb = glb, gub = gub)

    #---------
    #For Local Obj Tracking of Rubbles (no time guide)
    #---------
    # print("Normal Switching TIme limit")
    # #Switching Time Constraint
    # for phase_cnt in range(Nphase):
    #     if GaitPattern[phase_cnt] == 'InitialDouble':
    #         g.append(Ts[phase_cnt])
    #         glb.append(np.array([PhaseDuration_Limits["DoubleSupport_Min"]])) #old:0.1 - 0.3
    #         gub.append(np.array([PhaseDuration_Limits["DoubleSupport_Max"]]))
    #     elif GaitPattern[phase_cnt] == 'Swing':
    #         g.append(Ts[phase_cnt]-Ts[phase_cnt-1]) #0.6-1
    #         glb.append(np.array([PhaseDuration_Limits["SingleSupport_Min"]])) #old - 0.8-1.2
    #         gub.append(np.array([PhaseDuration_Limits["SingleSupport_Max"]])) 
    #     elif GaitPattern[phase_cnt] == 'DoubleSupport':
    #         g.append(Ts[phase_cnt]-Ts[phase_cnt-1])#0.05-0.3
    #         glb.append(np.array([PhaseDuration_Limits["DoubleSupport_Min"]]))
    #         gub.append(np.array([PhaseDuration_Limits["DoubleSupport_Max"]])) #0.1 - 0.3
    #     else:
    #         raise Exception("Unknown Phase Name")

    #---------
    #For Local Obj Tracking of Large Slope (need time guide)
    #---------
    if LocalObjMode == False:
        print("Normal Switching Time limit")
        #Switching Time Constraint
        for phase_cnt in range(Nphase):
            if GaitPattern[phase_cnt] == 'InitialDouble':
                g.append(Ts[phase_cnt])
                glb.append(np.array([PhaseDuration_Limits["DoubleSupport_Min"]])) #old:0.1 - 0.3
                gub.append(np.array([PhaseDuration_Limits["DoubleSupport_Max"]]))
            elif GaitPattern[phase_cnt] == 'Swing':
                g.append(Ts[phase_cnt]-Ts[phase_cnt-1]) #0.6-1
                glb.append(np.array([PhaseDuration_Limits["SingleSupport_Min"]])) #old - 0.8-1.2
                gub.append(np.array([PhaseDuration_Limits["SingleSupport_Max"]])) 
            elif GaitPattern[phase_cnt] == 'DoubleSupport':
                g.append(Ts[phase_cnt]-Ts[phase_cnt-1])#0.05-0.3
                glb.append(np.array([PhaseDuration_Limits["DoubleSupport_Min"]]))
                gub.append(np.array([PhaseDuration_Limits["DoubleSupport_Max"]])) #0.1 - 0.3
            else:
                raise Exception("Unknown Phase Name")
    elif LocalObjMode == True:
        print("Local Obj Switching Time Limit")
        #Timing Constraints (Slack Constrained)
        g, glb, gub = slackConstrained_SingleVar(a = Ts[0], b = InitDS_Ts_obj, slackratio = 0.15, g = g, glb = glb, gub = gub)
        g, glb, gub = slackConstrained_SingleVar(a = Ts[1]-Ts[0], b = SS_Ts_obj-InitDS_Ts_obj, slackratio = 0.15, g = g, glb = glb, gub = gub)
        g, glb, gub = slackConstrained_SingleVar(a = Ts[2]-Ts[1], b = DS_Ts_obj-SS_Ts_obj, slackratio = 0.15, g = g, glb = glb, gub = gub)

    # #Timing Constraints (Fixed)
    # g, glb, gub = std_eq_constraint(a = ca.vertcat(Ts[0],   Ts[1],   Ts[2]), 
    #                                 b = ca.vertcat(InitDS_Ts_obj, SS_Ts_obj, DS_Ts_obj), g = g, glb= glb, gub = gub)

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Variable Index - !!!
    #This is the pure Index, when try to get the array using other routines, we need to add "+1" at the last index due to Python indexing conventions
    #First set of variables start counting from 0, The enumeration of the last knot is N_K-1
    x_index = (0,N_K-1);                                y_index = (x_index[1]+1,x_index[1]+N_K);             z_index = (y_index[1]+1,y_index[1]+N_K)
    xdot_index = (z_index[1]+1,z_index[1]+N_K);         ydot_index = (xdot_index[1]+1,xdot_index[1]+N_K);    zdot_index = (ydot_index[1]+1,ydot_index[1]+N_K)
    Lx_index = (zdot_index[1]+1,zdot_index[1]+N_K);     Ly_index = (Lx_index[1]+1,Lx_index[1]+N_K);          Lz_index = (Ly_index[1]+1,Ly_index[1]+N_K)
    Ldotx_index = (Lz_index[1]+1,Lz_index[1]+N_K);      Ldoty_index = (Ldotx_index[1]+1,Ldotx_index[1]+N_K); Ldotz_index = (Ldoty_index[1]+1,Ldoty_index[1]+N_K)
    FL1x_index = (Ldotz_index[1]+1,Ldotz_index[1]+N_K); FL1y_index = (FL1x_index[1]+1,FL1x_index[1]+N_K);    FL1z_index = (FL1y_index[1]+1,FL1y_index[1]+N_K)
    FL2x_index = (FL1z_index[1]+1,FL1z_index[1]+N_K);   FL2y_index = (FL2x_index[1]+1,FL2x_index[1]+N_K);    FL2z_index = (FL2y_index[1]+1,FL2y_index[1]+N_K)
    FL3x_index = (FL2z_index[1]+1,FL2z_index[1]+N_K);   FL3y_index = (FL3x_index[1]+1,FL3x_index[1]+N_K);    FL3z_index = (FL3y_index[1]+1,FL3y_index[1]+N_K)
    FL4x_index = (FL3z_index[1]+1,FL3z_index[1]+N_K);   FL4y_index = (FL4x_index[1]+1,FL4x_index[1]+N_K);    FL4z_index = (FL4y_index[1]+1,FL4y_index[1]+N_K)
    FR1x_index = (FL4z_index[1]+1,FL4z_index[1]+N_K);   FR1y_index = (FR1x_index[1]+1,FR1x_index[1]+N_K);    FR1z_index = (FR1y_index[1]+1,FR1y_index[1]+N_K)
    FR2x_index = (FR1z_index[1]+1,FR1z_index[1]+N_K);   FR2y_index = (FR2x_index[1]+1,FR2x_index[1]+N_K);    FR2z_index = (FR2y_index[1]+1,FR2y_index[1]+N_K)
    FR3x_index = (FR2z_index[1]+1,FR2z_index[1]+N_K);   FR3y_index = (FR3x_index[1]+1,FR3x_index[1]+N_K);    FR3z_index = (FR3y_index[1]+1,FR3y_index[1]+N_K)
    FR4x_index = (FR3z_index[1]+1,FR3z_index[1]+N_K);   FR4y_index = (FR4x_index[1]+1,FR4x_index[1]+N_K);    FR4z_index = (FR4y_index[1]+1,FR4y_index[1]+N_K)
    px_index = (FR4z_index[1]+1,FR4z_index[1]+Nstep);   py_index = (px_index[1]+1,px_index[1]+Nstep);        pz_index = (py_index[1]+1,py_index[1]+Nstep)
    Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)

    var_index = {"x":x_index,          "y":y_index,          "z":z_index,        
                 "xdot":xdot_index,    "ydot":ydot_index,    "zdot":zdot_index,
                 "Lx":Lx_index,        "Ly":Ly_index,        "Lz":Lz_index,      
                 "Ldotx":Ldotx_index,  "Ldoty":Ldoty_index,  "Ldotz":Ldotz_index,
                 "FL1x":FL1x_index,    "FL1y":FL1y_index,    "FL1z":FL1z_index,
                 "FL2x":FL2x_index,    "FL2y":FL2y_index,    "FL2z":FL2z_index,
                 "FL3x":FL3x_index,    "FL3y":FL3y_index,    "FL3z":FL3z_index,
                 "FL4x":FL4x_index,    "FL4y":FL4y_index,    "FL4z":FL4z_index,
                 "FR1x":FR1x_index,    "FR1y":FR1y_index,    "FR1z":FR1z_index,
                 "FR2x":FR2x_index,    "FR2y":FR2y_index,    "FR2z":FR2z_index,
                 "FR3x":FR3x_index,    "FR3y":FR3y_index,    "FR3z":FR3z_index,
                 "FR4x":FR4x_index,    "FR4y":FR4y_index,    "FR4z":FR4z_index,
                 "px":px_index,        "py":py_index,        "pz":pz_index,
                 "Ts":Ts_index,
    }

    return DecisionVars, DecisionVars_lb, DecisionVars_ub, J, g, glb, gub, var_index

#Second Level, remember to make the max Ts duration to 3s
#Function to build the second level
#Nsteps: Number of steps in the second level, = Total Number of Steps of the Entire Lookahead Horizon - 1
def NLP_SecondLevel(m = 100.0, Nk_Local = 7, Nsteps = 1, AngularDynamics=True, ParameterList = None, PhaseDuration_Limits = None, miu = 0.3):
    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constant Parameters
    G = 9.80665 #kg/m^2
    #Friction Coefficient
    #   Force Limits
    F_bound = 400.0
    Fxlb = -F_bound; Fxub = F_bound
    Fylb = -F_bound; Fyub = F_bound
    Fzlb = -F_bound; Fzub = F_bound
    #   Angular Momentum Bounds
    L_bound = 5;    Llb = -L_bound;       Lub = L_bound
    Ldot_bound = 5; Ldotlb = -Ldot_bound; Ldotub = Ldot_bound
    #Lowest Z
    z_lowest = -5.0
    z_highest = 5.0
    #CoM Height with respect to Footstep Location (in the local stance frame, think about standstill pose)
    CoM_z_to_Foot_min = 0.6 #0.65 #0.6
    CoM_z_to_Foot_max = 0.8 #0.75
    #   Terrain Model
    #       Flat Terrain
    #TerrainNorm = [0,0,1] 
    #TerrainTangentX = [1,0,0]
    #TerrainTangentY = [0,1,0]

    #------------------------------------------------------------------------------------------------------------------------
    #Decide Motion Parameters
    #Gait Pattern
    GaitPattern = ["InitialDouble","Swing","DoubleSupport"] + ["InitialDouble", "Swing","DoubleSupport"]*(Nsteps-1) #,'RightSupport','DoubleSupport','LeftSupport','DoubleSupport'
    #   Number of Phases
    Nphase = len(GaitPattern)
    #   Compute Number of Total knots/ticks, but the enumeration start from 0 to N_K-1
    N_K = Nk_Local*Nphase + 1 #+1 the last knots to finalize the plan

    # #-----------------------------------------------------------------------------------------------------------------------
    # #Load kinematics Polytope
    # #   Not local than server
    # kinefilepath = "/home/jiayu/Desktop/multicontact_learning_local_objectives/python/kinematics_polytope/kinematics_constraints.p"
    # if os.path.isfile(kinefilepath) == False:
    #     kinefilepath = "/afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/kinematics_polytope/kinematics_constraints.p"

    # with open(kinefilepath, 'rb') as f:
    #     kinematics_constraints= pickle.load(f)
    
    # #CoM Polytopes
    # K_CoM_Right = kinematics_constraints["K_CoM_in_Right_Contact"];     k_CoM_Right = kinematics_constraints["k_CoM_in_Right_Contact"]
    # K_CoM_Left  = kinematics_constraints["K_CoM_in_Left_Contact"];      k_CoM_Left  = kinematics_constraints["k_CoM_in_Left_Contact"]

    # #Relative Footstep constraints
    # Q_rf_in_lf = kinematics_constraints["Q_Right_Contact_in_Left_Contact"];    q_rf_in_lf = kinematics_constraints["q_Right_Contact_in_Left_Contact"]
    # Q_lf_in_rf = kinematics_constraints["Q_Left_Contact_in_Right_Contact"];    q_lf_in_rf = kinematics_constraints["q_Left_Contact_in_Right_Contact"]

    #Get Kinematics Constraint for Talos
    #CoM kinematics constraint, give homogenous transformaiton (the last column seems like dont make a diff)
    K_CoM_Right,k_CoM_Right = right_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    K_CoM_Left,k_CoM_Left = left_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    #Another way
    #kinematicConstraints = genKinematicConstraints(left_foot_constraints,right_foot_constraints)
    #K_CoM_Left = kinematicConstraints[0][0];   k_CoM_Left = kinematicConstraints[0][1]
    #K_CoM_Right = kinematicConstraints[1][0];  k_CoM_Right = kinematicConstraints[1][1]
    
    #Relative Foot Constraint matrices
    #Relative foot constraint, give homogenous transformation (the last column seems like dont make a diff)
    Q_rf_in_lf,q_rf_in_lf = right_foot_in_lf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    Q_lf_in_rf,q_lf_in_rf = left_foot_in_rf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    #Another way
    #relativeConstraints = genFootRelativeConstraints(right_foot_in_lf_frame_constraints,left_foot_in_rf_frame_constraints)
    #Q_rf_in_lf = relativeConstraints[0][0];   q_rf_in_lf = relativeConstraints[0][1] #named lf in rf, but representing rf in lf
    #Q_lf_in_rf = relativeConstraints[1][0];   q_lf_in_rf = relativeConstraints[1][1] #named rf in lf, but representing lf in rf
    #-----------------------------------------------------------------------------------------------------------------------

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Casadi Parameters
    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ParameterList["LeftSwingFlag"]
    ParaRightSwingFlag = ParameterList["RightSwingFlag"]
    
    #Initial Left Foot Contact Location
    PLx_init = ParameterList["PLx_init"];   PLy_init = ParameterList["PLy_init"];   PLz_init = ParameterList["PLz_init"]
    PL_init = ca.vertcat(PLx_init,PLy_init,PLz_init)
    #Initial Right Foot Contact Location
    PRx_init = ParameterList["PRx_init"];   PRy_init = ParameterList["PRy_init"];   PRz_init = ParameterList["PRz_init"]
    PR_init = ca.vertcat(PRx_init,PRy_init,PRz_init)

    #Surface Patches
    SurfParas = ParameterList["SurfParas"]

    #Tangents and Norms
    #Initial Contact Norm and Tangents
    PL_init_Norm = ParameterList["PL_init_Norm"];   PL_init_TangentX = ParameterList["PL_init_TangentX"];   PL_init_TangentY = ParameterList["PL_init_TangentY"]
    PR_init_Norm = ParameterList["PR_init_Norm"];   PR_init_TangentX = ParameterList["PR_init_TangentX"];   PR_init_TangentY = ParameterList["PR_init_TangentY"]

    #Future Contact Norm and Tangents
    SurfNorms = ParameterList["SurfNorms"]                
    SurfTangentsX = ParameterList["SurfTangentsX"]
    SurfTangentsY = ParameterList["SurfTangentsY"]

    #Surface Orientations
    #Initial Contact Surface Orientations
    PL_init_Orientation = ParameterList["PL_init_Orientation"];  PL_init_Orientation = ca.reshape(PL_init_Orientation,3,3).T
    PR_init_Orientation = ParameterList["PR_init_Orientation"];  PR_init_Orientation = ca.reshape(PR_init_Orientation,3,3).T
    #
    SurfOriens = ParameterList["SurfOrientations"]

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Variables and Bounds, Parameters
    #   CoM Position x, y, z
    x = ca.SX.sym('x',N_K);   x_lb = np.array([[0.0]*(x.shape[0]*x.shape[1])]);         x_ub = np.array([[50.0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    y = ca.SX.sym('y',N_K);   y_lb = np.array([[-1.0]*(y.shape[0]*y.shape[1])]);        y_ub = np.array([[1.0]*(y.shape[0]*y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    z = ca.SX.sym('z',N_K);   z_lb = np.array([[z_lowest]*(z.shape[0]*z.shape[1])]);  z_ub = np.array([[z_highest]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #   CoM Velocity x, y, z
    xdot = ca.SX.sym('xdot',N_K);   xdot_lb = np.array([[0.0]*(xdot.shape[0]*xdot.shape[1])]);   xdot_ub = np.array([[1.5]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    ydot = ca.SX.sym('ydot',N_K);   ydot_lb = np.array([[-1.5]*(ydot.shape[0]*ydot.shape[1])]);   ydot_ub = np.array([[1.5]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    zdot = ca.SX.sym('zdot',N_K);   zdot_lb = np.array([[-1.5]*(zdot.shape[0]*zdot.shape[1])]);   zdot_ub = np.array([[1.5]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #   Angular Momentum x, y ,z
    Lx = ca.SX.sym('Lx',N_K);       Lx_lb = np.array([[Llb]*(Lx.shape[0]*Lx.shape[1])]);    Lx_ub = np.array([[Lub]*(Lx.shape[0]*Lx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ly = ca.SX.sym('Ly',N_K);       Ly_lb = np.array([[Llb]*(Ly.shape[0]*Ly.shape[1])]);    Ly_ub = np.array([[Lub]*(Ly.shape[0]*Ly.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Lz = ca.SX.sym('Lz',N_K);       Lz_lb = np.array([[Llb]*(Lz.shape[0]*Lz.shape[1])]);    Lz_ub = np.array([[Lub]*(Lz.shape[0]*Lz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #   Angular Momentum rate x, y, z
    Ldotx = ca.SX.sym('Ldotx',N_K); Ldotx_lb = np.array([[Ldotlb]*(Ldotx.shape[0]*Ldotx.shape[1])]); Ldotx_ub = np.array([[Ldotub]*(Ldotx.shape[0]*Ldotx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldoty = ca.SX.sym('Ldoty',N_K); Ldoty_lb = np.array([[Ldotlb]*(Ldoty.shape[0]*Ldoty.shape[1])]); Ldoty_ub = np.array([[Ldotub]*(Ldoty.shape[0]*Ldoty.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    Ldotz = ca.SX.sym('Ldotz',N_K); Ldotz_lb = np.array([[Ldotlb]*(Ldotz.shape[0]*Ldotz.shape[1])]); Ldotz_ub = np.array([[Ldotub]*(Ldotz.shape[0]*Ldotz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Left Foot Forces
    #Left Foot Contact Point 1 x, y, z
    FL1x = ca.SX.sym('FL1x',N_K);   FL1x_lb = np.array([[Fxlb]*(FL1x.shape[0]*FL1x.shape[1])]);   FL1x_ub = np.array([[Fxub]*(FL1x.shape[0]*FL1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1y = ca.SX.sym('FL1y',N_K);   FL1y_lb = np.array([[Fylb]*(FL1y.shape[0]*FL1y.shape[1])]);   FL1y_ub = np.array([[Fyub]*(FL1y.shape[0]*FL1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1z = ca.SX.sym('FL1z',N_K);   FL1z_lb = np.array([[Fzlb]*(FL1z.shape[0]*FL1z.shape[1])]);   FL1z_ub = np.array([[Fzub]*(FL1z.shape[0]*FL1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Left Foot Contact Point 2 x, y, z
    FL2x = ca.SX.sym('FL2x',N_K);   FL2x_lb = np.array([[Fxlb]*(FL2x.shape[0]*FL2x.shape[1])]);   FL2x_ub = np.array([[Fxub]*(FL2x.shape[0]*FL2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2y = ca.SX.sym('FL2y',N_K);   FL2y_lb = np.array([[Fylb]*(FL2y.shape[0]*FL2y.shape[1])]);   FL2y_ub = np.array([[Fyub]*(FL2y.shape[0]*FL2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2z = ca.SX.sym('FL2z',N_K);   FL2z_lb = np.array([[Fzlb]*(FL2z.shape[0]*FL2z.shape[1])]);   FL2z_ub = np.array([[Fzub]*(FL2z.shape[0]*FL2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Left Foot Contact Point 3 x, y, z
    FL3x = ca.SX.sym('FL3x',N_K);   FL3x_lb = np.array([[Fxlb]*(FL3x.shape[0]*FL3x.shape[1])]);   FL3x_ub = np.array([[Fxub]*(FL3x.shape[0]*FL3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3y = ca.SX.sym('FL3y',N_K);   FL3y_lb = np.array([[Fylb]*(FL3y.shape[0]*FL3y.shape[1])]);   FL3y_ub = np.array([[Fyub]*(FL3y.shape[0]*FL3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3z = ca.SX.sym('FL3z',N_K);   FL3z_lb = np.array([[Fzlb]*(FL3z.shape[0]*FL3z.shape[1])]);   FL3z_ub = np.array([[Fzub]*(FL3z.shape[0]*FL3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Left Foot Contact Point 4 x-axis
    FL4x = ca.SX.sym('FL4x',N_K);   FL4x_lb = np.array([[Fxlb]*(FL4x.shape[0]*FL4x.shape[1])]);   FL4x_ub = np.array([[Fxub]*(FL4x.shape[0]*FL4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4y = ca.SX.sym('FL4y',N_K);   FL4y_lb = np.array([[Fylb]*(FL4y.shape[0]*FL4y.shape[1])]);   FL4y_ub = np.array([[Fyub]*(FL4y.shape[0]*FL4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4z = ca.SX.sym('FL4z',N_K);   FL4z_lb = np.array([[Fzlb]*(FL4z.shape[0]*FL4z.shape[1])]);   FL4z_ub = np.array([[Fzub]*(FL4z.shape[0]*FL4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    
    #Right Contact Force x-axis
    #Right Foot Contact Point 1 x, y, z
    FR1x = ca.SX.sym('FR1x',N_K);   FR1x_lb = np.array([[Fxlb]*(FR1x.shape[0]*FR1x.shape[1])]);   FR1x_ub = np.array([[Fxub]*(FR1x.shape[0]*FR1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1y = ca.SX.sym('FR1y',N_K);   FR1y_lb = np.array([[Fylb]*(FR1y.shape[0]*FR1y.shape[1])]);   FR1y_ub = np.array([[Fyub]*(FR1y.shape[0]*FR1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1z = ca.SX.sym('FR1z',N_K);   FR1z_lb = np.array([[Fzlb]*(FR1z.shape[0]*FR1z.shape[1])]);   FR1z_ub = np.array([[Fzub]*(FR1z.shape[0]*FR1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Right Foot Contact Point 2 x, y, z
    FR2x = ca.SX.sym('FR2x',N_K);   FR2x_lb = np.array([[Fxlb]*(FR2x.shape[0]*FR2x.shape[1])]);   FR2x_ub = np.array([[Fxub]*(FR2x.shape[0]*FR2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2y = ca.SX.sym('FR2y',N_K);   FR2y_lb = np.array([[Fylb]*(FR2y.shape[0]*FR2y.shape[1])]);   FR2y_ub = np.array([[Fyub]*(FR2y.shape[0]*FR2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2z = ca.SX.sym('FR2z',N_K);   FR2z_lb = np.array([[Fzlb]*(FR2z.shape[0]*FR2z.shape[1])]);   FR2z_ub = np.array([[Fzub]*(FR2z.shape[0]*FR2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Right Foot Contact Point 3 x, y, z
    FR3x = ca.SX.sym('FR3x',N_K);   FR3x_lb = np.array([[Fxlb]*(FR3x.shape[0]*FR3x.shape[1])]);   FR3x_ub = np.array([[Fxub]*(FR3x.shape[0]*FR3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3y = ca.SX.sym('FR3y',N_K);   FR3y_lb = np.array([[Fylb]*(FR3y.shape[0]*FR3y.shape[1])]);   FR3y_ub = np.array([[Fyub]*(FR3y.shape[0]*FR3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3z = ca.SX.sym('FR3z',N_K);   FR3z_lb = np.array([[Fzlb]*(FR3z.shape[0]*FR3z.shape[1])]);   FR3z_ub = np.array([[Fzub]*(FR3z.shape[0]*FR3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Right Foot Contact Point 4 x, y, z
    FR4x = ca.SX.sym('FR4x',N_K);   FR4x_lb = np.array([[Fxlb]*(FR4x.shape[0]*FR4x.shape[1])]);   FR4x_ub = np.array([[Fxub]*(FR4x.shape[0]*FR4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4y = ca.SX.sym('FR4y',N_K);   FR4y_lb = np.array([[Fylb]*(FR4y.shape[0]*FR4y.shape[1])]);   FR4y_ub = np.array([[Fyub]*(FR4y.shape[0]*FR4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4z = ca.SX.sym('FR4z',N_K);   FR4z_lb = np.array([[Fzlb]*(FR4z.shape[0]*FR4z.shape[1])]);   FR4z_ub = np.array([[Fzub]*(FR4z.shape[0]*FR4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements

    #Initial Contact Location (First step made in the first level), need to connect to the first level landing location 
    #   Px, Py, Pz
    px_init = ca.SX.sym('px_init');   px_init_lb = np.array([-1.0]);   px_init_ub = np.array([50.0])
    py_init = ca.SX.sym('py_init');   py_init_lb = np.array([-2.0]);   py_init_ub = np.array([2.0])
    pz_init = ca.SX.sym('pz_init');   pz_init_lb = np.array([-5.0]);   pz_init_ub = np.array([5.0])

    #   Contact Location Sequence
    px = [];   px_lb = [];   px_ub = []
    py = [];   py_lb = [];   py_ub = []
    pz = [];   pz_lb = [];   pz_ub = []

    for stepIdx in range(Nsteps):
        #Nsteps: Number of steps in the second level, = Total Number of Steps of the Entire Lookahead Horizon - 1
        #Therefore the enumeration of contact location sequences start counting from 1 (to be aligned with step number in the entire horizon)
        pxtemp = ca.SX.sym('px'+str(stepIdx + 1));   px.append(pxtemp);   px_lb.append(np.array([-1.0]));   px_ub.append(np.array([50.0]))
        pytemp = ca.SX.sym('py'+str(stepIdx + 1));   py.append(pytemp);   py_lb.append(np.array([-2.0]));   py_ub.append(np.array([2.0]))
        pztemp = ca.SX.sym('pz'+str(stepIdx + 1));   pz.append(pztemp);   pz_lb.append(np.array([-5.0]));   pz_ub.append(np.array([5.0]))

    #Switching Time Vector
    Ts = [];   Ts_lb = [];   Ts_ub = []
    for n_phase in range(Nphase):
        #Ts start counting from 1, Ts0 = 0
        Tstemp = ca.SX.sym('Ts'+str(n_phase+1));   Ts.append(Tstemp); 
        #Define switching time limits based on phase duration limits  
        Ts_lb.append(np.array([PhaseDuration_Limits["DoubleSupport_Min"]*0.1]));   
        Ts_ub.append(np.array([(PhaseDuration_Limits["DoubleSupport_Max"]+PhaseDuration_Limits["SingleSupport_Max"]+PhaseDuration_Limits["DoubleSupport_Max"])*(Nphase+1)]))    
    
    #   Collect all Decision Variables
    DecisionVars = ca.vertcat(x,       y,       z,    xdot,  ydot,  zdot,
                              Lx,      Ly,      Lz,   Ldotx, Ldoty, Ldotz,
                              FL1x,    FL1y,    FL1z, FL2x,  FL2y,  FL2z,  FL3x,  FL3y,  FL3z,   FL4x,  FL4y,  FL4z,
                              FR1x,    FR1y,    FR1z, FR2x,  FR2y,  FR2z,  FR3x,  FR3y,  FR3z,   FR4x,  FR4y,  FR4z,
                              px_init, py_init, pz_init,
                              *px,     *py,     *pz,
                              *Ts)
    #print(DecisionVars)
    DecisionVarsShape = DecisionVars.shape

    #   Collect all lower bound and upper bound
    DecisionVars_lb = (x_lb,     y_lb,     z_lb,     xdot_lb,  ydot_lb,  zdot_lb,
                       Lx_lb,    Ly_lb,    Lz_lb,    Ldotx_lb, Ldoty_lb, Ldotz_lb,
                       FL1x_lb,  FL1y_lb,  FL1z_lb,  FL2x_lb,  FL2y_lb,  FL2z_lb,  FL3x_lb,  FL3y_lb,  FL3z_lb,  FL4x_lb,  FL4y_lb,  FL4z_lb,
                       FR1x_lb,  FR1y_lb,  FR1z_lb,  FR2x_lb,  FR2y_lb,  FR2z_lb,  FR3x_lb,  FR3y_lb,  FR3z_lb,  FR4x_lb,  FR4y_lb,  FR4z_lb,
                       px_init_lb, py_init_lb, pz_init_lb,
                       px_lb,      py_lb,      pz_lb,
                       Ts_lb)
    DecisionVars_lb = np.concatenate(DecisionVars_lb,axis=None)

    DecisionVars_ub = (x_ub,      y_ub,       z_ub,       xdot_ub,     ydot_ub,     zdot_ub,
                      Lx_ub,      Ly_ub,      Lz_ub,      Ldotx_ub,    Ldoty_ub,    Ldotz_ub,
                      FL1x_ub,    FL1y_ub,    FL1z_ub,    FL2x_ub,     FL2y_ub,     FL2z_ub,   FL3x_ub,   FL3y_ub,   FL3z_ub,  FL4x_ub,   FL4y_ub,   FL4z_ub,
                      FR1x_ub,    FR1y_ub,    FR1z_ub,    FR2x_ub,     FR2y_ub,     FR2z_ub,   FR3x_ub,   FR3y_ub,   FR3z_ub,  FR4x_ub,   FR4y_ub,   FR4z_ub,
                      px_init_ub, py_init_ub, pz_init_ub, 
                      px_ub,      py_ub,      pz_ub,
                      Ts_ub)
    DecisionVars_ub = np.concatenate(DecisionVars_ub,axis=None)

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constrains and Running Cost
    g = [];   glb = [];   gub = []
    J = 0

    #Time Span Setup
    tau_upper_limit = 1.0
    tauStepLength = tau_upper_limit/(N_K-1) #Get the interval length, total number of knots - 1

    #Constraints for all knots
    #Loop over all Phases (Knots)
    for Nph in range(Nphase):
        #Decide Number of Knots
        if Nph == Nphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local  

        #Decide Time Vector
        if Nph == 0: #first phase
            h = tauStepLength*Nphase*(Ts[Nph]-0)
        else: #other phases
            h = tauStepLength*Nphase*(Ts[Nph]-Ts[Nph-1])

        for Local_k_Count in range(Nk_ThisPhase):
            #Get knot number across the entire time line
            k = Nph*Nk_Local + Local_k_Count
            #------------------------------------------
            #Build useful vectors
            #   Forces
            FL1_k = ca.vertcat(FL1x[k],FL1y[k],FL1z[k]);   FL2_k = ca.vertcat(FL2x[k],FL2y[k],FL2z[k]);   FL3_k = ca.vertcat(FL3x[k],FL3y[k],FL3z[k]);   FL4_k = ca.vertcat(FL4x[k],FL4y[k],FL4z[k])
            FR1_k = ca.vertcat(FR1x[k],FR1y[k],FR1z[k]);   FR2_k = ca.vertcat(FR2x[k],FR2y[k],FR2z[k]);   FR3_k = ca.vertcat(FR3x[k],FR3y[k],FR3z[k]);   FR4_k = ca.vertcat(FR4x[k],FR4y[k],FR4z[k])
            #   CoM
            CoM_k = ca.vertcat(x[k],y[k],z[k])
            #   Ldot
            Ldot_k = ca.vertcat(Ldotx[k],Ldoty[k],Ldotz[k])
            
            #-------------------------------------------
            #Phase dependent Constraints (CoM Kinematics and Angular Dynamics)
            #Get Step Counter
            StepCnt = Nph//3
            
            if GaitPattern[Nph]=='InitialDouble':
                #Special Case:
                if StepCnt == 0: #The first phase in the First STEP (Initial Double, need special care)
                    #initial support foot (the landing foot from the first phase)
                    p_init = ca.vertcat(px_init,py_init,pz_init)
                    p_init_TangentX = SurfTangentsX[0:3]
                    p_init_TangentY = SurfTangentsY[0:3]
                    p_init_Norm = SurfNorms[0:3]
                    p_init_Orientation = ca.reshape(SurfOriens[0:9],3,3).T

                    #-----------
                    #Case 1
                    #If First Level Swing the Left, the the 0 phase (InitDouble) has p_init as the left support, PR_init as the right support
                    #(same as the double support phase of the first step)-> Left foot Moved (p_init), Right Foot stay stationary (PR_init)
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Kinematics Constraint
                    #CoM in Left (p_init)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_init, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                                 ContactFrameOrientation = p_init_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    #CoM in Right (PR_init)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = PR_init, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                                 ContactFrameOrientation = PR_init_Orientation,
                                                 g = g, glb = glb, gub = gub)
                
                    #CoM Height Constraint (Left p_init foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max,
                                                           ContactFrameOrientation = p_init_Orientation,
                                                           g = g, glb = glb, gub = gub)
                    #CoM Height Constraint (Right foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PR_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = PR_init_Orientation,
                                                           g = g, glb = glb, gub = gub)
                    
                    #Angular Dynamics
                    if AngularDynamics == True:
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator = SwingLegFlag, 
                                                                              Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                              PL = p_init, PL_TangentX = p_init_TangentX, PL_TangentY = p_init_TangentY, 
                                                                              PR = PR_init, PR_TangentX = PR_init_TangentX, PR_TangentY = PR_init_TangentY, 
                                                                              FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, 
                                                                              FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraint
                    #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the unilateral constraint on p_init
                    #Left Foot force (p_init)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the PR_init
                    #Right Foot (PR_init)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the friction cone constraint on p_init
                    #Left Foot (p_init)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the PR_init
                    #Right Foot (PR_init)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    
                    #-------------
                    #Case 2
                    #If First Level Swing the Right, the the 0 phase (InitDouble) has p_init as the Right support, PL_init as the Left support
                    #(same as the double support phase of the first step)-> Right foot Moved (p_init), Left Foot stay stationary (PL_init)
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Kinematics Constraint
                    #CoM in Left (PL_init)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = PL_init, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                                 ContactFrameOrientation = PL_init_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    #CoM in Right (p_init)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_init, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                                 ContactFrameOrientation = p_init_Orientation,
                                                 g = g, glb = glb, gub = gub)
                
                    #CoM Height Constraint (Left PL_init foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PL_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = PL_init_Orientation,
                                                           g = g, glb = glb, gub = gub)
                    #CoM Height Constraint (Right p_init foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = p_init_Orientation,
                                                           g = g, glb = glb, gub = gub)
                    
                    #Angular Dynamics
                    if AngularDynamics == True:
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator = SwingLegFlag, 
                                                                              Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                              PL = PL_init, PL_TangentX = PL_init_TangentX, PL_TangentY = PL_init_TangentY, 
                                                                              PR = p_init,  PR_TangentX = p_init_TangentX,  PR_TangentY = p_init_TangentY, 
                                                                              FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, 
                                                                              FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraint
                    #Left Foot (PL_init)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                    #Right Foot (p_init)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    
                    #Friction Cone Constraint
                    #Left Foot (PL_init)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #Right Foot (p_init)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)

                elif StepCnt > 0:#Other Cases
                    #Get contact location and Terrain Tangents and Norms
                    if StepCnt == 1: #Step 1 needs special care (NOTE: Step Count Start from 0)
                        #Previous Step
                        p_previous = ca.vertcat(px_init,py_init,pz_init)
                        p_previous_TangentX = SurfTangentsX[0:3];   p_previous_TangentY = SurfTangentsY[0:3]
                        p_previous_Norm = SurfNorms[0:3]
                        p_previous_Orientation = ca.reshape(SurfOriens[0:9],3,3).T

                        #Current Step
                        #In second level, Surfaces index is Step Vector Index (for px, py, pz, here is StepCnt-1) + 1
                        #For Intial Double Support, previous step is StepNum - 2, current step is StepNum - 1
                        #but index for the tangents and norm need to +1 as StepNum (in the second level) + 1 = StepNum in the entire horizon
                        p_current = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_current_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3];   p_current_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3] 
                        p_current_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]
                        p_current_Orientation = ca.reshape(SurfOriens[StepCnt*9:StepCnt*9+9],3,3).T

                    else: #Like Step 2, 3, 4 .....
                        #For Intial Double Support, previous step is StepNum - 2, current step is StepNum - 1
                        #but index for the tangents and norm need to +1 as StepNum (in the second level) + 1 = StepNum in the entire horizon
                        #For Initial Double Support, the contact config is the same as the double support phase of the previous step, where p_current is the landed foot
                        #p_previous is the non-moving foot
                        p_previous = ca.vertcat(px[StepCnt-2],py[StepCnt-2],pz[StepCnt-2])
                        p_previous_TangentX = SurfTangentsX[(StepCnt-1)*3:(StepCnt-1)*3+3];   p_previous_TangentY = SurfTangentsY[(StepCnt-1)*3:(StepCnt-1)*3+3]
                        p_previous_Norm = SurfNorms[(StepCnt-1)*3:(StepCnt-1)*3+3]
                        p_previous_Orientation = ca.reshape(SurfOriens[(StepCnt-1)*9:(StepCnt-1)*9+9],3,3).T

                        p_current = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_current_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3];   p_current_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                        p_current_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]
                        p_current_Orientation = ca.reshape(SurfOriens[StepCnt*9:StepCnt*9+9],3,3).T


                    #Give Constraint according to even and odd steps
                    if StepCnt%2 == 0: #Even Numbers of Footsteps
                        #----------
                        #Case 1
                        #If the first level swing the Left, then the Even Number of Steps in the Intial Double support phase have p_current as Left Support (Landed), p_previous as Right Support (Stationary)
                        #(same as the double support phase of the first step)->Left foot Moved (p_current), Right Foot stay stationary p_previous)
                        SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                        #CoM in the Left (p_current)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                     CoM_k = CoM_k, P = p_current, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                                     ContactFrameOrientation = p_current_Orientation,
                                                     g = g, glb = glb, gub = gub)
                        #CoM in the Right (p_previous)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                     CoM_k = CoM_k, P = p_previous, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                                     ContactFrameOrientation = p_previous_Orientation,
                                                     g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Left p_current foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                               ContactFrameOrientation = p_current_Orientation,
                                                               g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Right p_previous foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                               ContactFrameOrientation = p_previous_Orientation,
                                                               g = g, glb = glb, gub = gub)
                        #Angular Dynamics
                        if AngularDynamics == True:
                            if k<N_K-1:
                                g, glb, gub = Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator = SwingLegFlag, 
                                                                                    Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                                    PL = p_current, PL_TangentX = p_current_TangentX, PL_TangentY = p_current_TangentY, 
                                                                                    PR = p_previous, PR_TangentX = p_previous_TangentX, PR_TangentY = p_previous_TangentY, 
                                                                                    FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, 
                                                                                    FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k, g = g, glb = glb, gub = gub)
                        #Unilateral Constraint
                        #Left Foot (p_current)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        #Right Foot (p_previous)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        #Friction Cone Constraint
                        #Left Foot (p_current)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        #Right Foot (p_previous)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        
                        #---------
                        #Case 2
                        #If the first level swing the Right, then the Even Number of Steps in the Intial Double support phase have p_current as Right Support (Landed), 
                        #(same as the double support phase of the first step) -> Right Moved (p_current), Left stationary (p_previous)
                        SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                        #CoM in the Left (p_previous)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                     CoM_k = CoM_k, P = p_previous, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                                     ContactFrameOrientation = p_previous_Orientation,
                                                     g = g, glb = glb, gub = gub)
                        #CoM in the Right (p_current)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                     CoM_k = CoM_k, P = p_current, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                                     ContactFrameOrientation = p_current_Orientation,
                                                     g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Left p_previous foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                               ContactFrameOrientation = p_previous_Orientation,
                                                               g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Right p_current foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                               ContactFrameOrientation = p_current_Orientation,
                                                               g = g, glb = glb, gub = gub)
                        #Angular Dynamics
                        if AngularDynamics == True:
                            if k<N_K-1:
                                g, glb, gub = Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator = SwingLegFlag, 
                                                                                    Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                                    PL = p_previous, PL_TangentX = p_previous_TangentX, PL_TangentY = p_previous_TangentY, 
                                                                                    PR = p_current,  PR_TangentX = p_current_TangentX,  PR_TangentY = p_current_TangentY, 
                                                                                    FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, 
                                                                                    FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k, g = g, glb = glb, gub = gub)
                        #Unilateral Constraint
                        #Left foot (p_previous)                         
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        #Right foot (p_current)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        #Friction Cone Constraint
                        #Left foot (p_previous)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        #Right Foot (p_current)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)

                    elif StepCnt%2 == 1: #Odd Number of Steps
                        #---------
                        #Case 1
                        #If the first level swing the Left, then the Odd Number of Steps in the Intial Double support phase (Swing Right) have p_current as Right Support (Landed), p_previous as Left Support (Stationary)
                        #(same as the double support phase of the first step) -> Right Moved (p_current), Left Stay Fixed (p_previous)
                        SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                        #CoM in the Left (p_previous)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                     CoM_k = CoM_k, P = p_previous, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                                     ContactFrameOrientation = p_previous_Orientation,
                                                     g = g, glb = glb, gub = gub)
                        #CoM in the Right (p_current)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                     CoM_k = CoM_k, P = p_current, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                                     ContactFrameOrientation = p_current_Orientation,
                                                     g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Left p_previous foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                               ContactFrameOrientation = p_previous_Orientation,
                                                               g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Right p_current foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                               ContactFrameOrientation = p_current_Orientation,
                                                               g = g, glb = glb, gub = gub)
                        #Angular Dynamics
                        if AngularDynamics == True:
                            if k<N_K-1:
                                g, glb, gub = Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator = SwingLegFlag, 
                                                                                    Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                                    PL = p_previous, PL_TangentX = p_previous_TangentX, PL_TangentY = p_previous_TangentY, 
                                                                                    PR = p_current,  PR_TangentX = p_current_TangentX,  PR_TangentY = p_current_TangentY, 
                                                                                    FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, 
                                                                                    FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k, g = g, glb = glb, gub = gub)
                        #Unilateral Constraint
                        #Left foot (p_previous)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        #Right foot (p_current)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        #Friction Cone Constraint
                        #Left foot (p_previous)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        #Right Foot (p_current)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        
                        #-----
                        #Case 2
                        #If the first level swing the Right, then the Odd Number of Steps in the Intial Double support phase have p_current as Left Support (Landed), 
                        #(same as the double support phase of the first step) -> Left Moved (p_current), Right stationary (p_previous)
                        SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                        #CoM in the Left (p_current)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                     CoM_k = CoM_k, P = p_current, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                                     ContactFrameOrientation = p_current_Orientation,
                                                     g = g, glb = glb, gub = gub)
                        #CoM in the Right (p_previous)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                     CoM_k = CoM_k, P = p_previous, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                                     ContactFrameOrientation = p_previous_Orientation,
                                                     g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Left p_current foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                               ContactFrameOrientation = p_current_Orientation,
                                                               g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Right p_previous foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                               ContactFrameOrientation = p_previous_Orientation,
                                                               g = g, glb = glb, gub = gub)
                        #Angular Dynamics
                        if AngularDynamics == True:
                            if k<N_K-1:
                                g, glb, gub = Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator = SwingLegFlag, 
                                                                                    Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                                    PL = p_current, PL_TangentX = p_current_TangentX, PL_TangentY = p_current_TangentY, 
                                                                                    PR = p_previous,  PR_TangentX = p_previous_TangentX,  PR_TangentY = p_previous_TangentY, 
                                                                                    FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, 
                                                                                    FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k, g = g, glb = glb, gub = gub)
                        #Unilateral Constraint
                        #Left foot (p_current)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        #Right foot (p_previous)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        #Friction Cone Constraint
                        #Left foot (p_current)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        #right foot (p_previous)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)

            elif GaitPattern[Nph]== 'Swing':
                #Get contact location
                #In the swing phase, the stance leg is the landing foot of the previous step (Step Number - 1), 
                #but index for the tangents and norm need to +1 as StepNum (in the second level) + 1 = StepNum in the entire horizon
                if StepCnt == 0:#Special Case for the First Step (NOTE:Step 0)
                    p_stance = ca.vertcat(px_init,py_init,pz_init)
                    p_stance_TangentX = SurfTangentsX[0:3]
                    p_stance_TangentY = SurfTangentsY[0:3]
                    p_stance_Norm = SurfNorms[0:3]
                    p_stance_Orientation = ca.reshape(SurfOriens[0:9],3,3).T

                else: #For other Steps, indexed as 1,2,3,4
                    p_stance = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                    p_stance_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                    p_stance_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                    p_stance_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]
                    p_stance_Orientation = ca.reshape(SurfOriens[StepCnt*9:StepCnt*9+9],3,3).T

                #Give Constraint according to even and odd steps
                if StepCnt%2 == 0: #Even Number of Steps
                    #------
                    #Case 1
                    #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Left foot is the stance foot

                    #Kinematics Constraint
                    #CoM in the Left (p_stance)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_stance, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                                 ContactFrameOrientation = p_stance_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    #CoM Height Constraint (Left p_stance foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = p_stance_Orientation,
                                                           g = g, glb = glb, gub = gub)
                    #Angular Dynamics (Left Stance)
                    if AngularDynamics == True:
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_Swing(SwingLegIndicator = SwingLegFlag, 
                                                                      Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                      P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY,
                                                                      F1_k = FL1_k, F2_k = FL2_k, F3_k = FL3_k, F4_k = FL4_k, g = g, glb = glb, gub = gub)
    
                    #Zero Forces (Right Foot)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraints on Left Foot p_stance
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint on Left Foot p_stance
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)

                    #--------------
                    #Case 2
                    #If First Level Swing the Right, then the second level Even Number Phases (the first Phase) Swing the Left
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Right foot is the stance foot

                    #Kinematics Constraint
                    #CoM in the Right (p_stance)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_stance, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                                 ContactFrameOrientation = p_stance_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    #CoM Height Constraint (Right p_stance foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                 ContactFrameOrientation = p_stance_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    #Angular Dynamics (Right Stance)
                    if AngularDynamics == True:
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_Swing(SwingLegIndicator = SwingLegFlag, 
                                                                      Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                      P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY,
                                                                      F1_k = FR1_k, F2_k = FR2_k, F3_k = FR3_k, F4_k = FR4_k, g = g, glb = glb, gub = gub)
    
                    #Zero Forces (Left Foot)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraints on Right Foot p_stance
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint on Right Foot p_stance
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)

                elif StepCnt%2 == 1: #Odd Number of Steps
                    #--------
                    #Case 1
                    #If First Level Swing the Left, then the second level Odd Number Steps Swing the Left
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Right Foot is the stance foot

                    #Kinematics Constraint
                    #CoM in the Right (p_stance)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_stance, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                                 ContactFrameOrientation = p_stance_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_stance foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = p_stance_Orientation,
                                                           g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Right Stance)
                    if AngularDynamics == True:
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_Swing(SwingLegIndicator = SwingLegFlag, 
                                                                      Ldot_k = Ldot_k, CoM_k = CoM_k,
                                                                      P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, 
                                                                      F1_k = FR1_k, F2_k = FR2_k, F3_k = FR3_k, F4_k = FR4_k, g = g, glb = glb, gub = gub)

                    #Zero Forces (Left Foot)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraints on Right Foot p_stance
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint on Right Foot p_stance
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)

                    #-------
                    #Case 2
                    #If First Level Swing the Right, then the second level Odd Number Steps Swing the Right
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Left foot is the stance foot

                    #Kinematics Constraint
                    #CoM in the Left (p_stance)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_stance, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                                 ContactFrameOrientation = p_stance_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Left p_stance foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = p_stance_Orientation,
                                                           g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Left Stance)
                    if AngularDynamics == True:
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_Swing(SwingLegIndicator = SwingLegFlag, 
                                                                      Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                      P = p_stance, P_TangentX = p_stance_TangentX, P_TangentY = p_stance_TangentY, 
                                                                      F1_k = FL1_k, F2_k = FL2_k, F3_k = FL3_k, F4_k = FL4_k, g = g, glb = glb, gub = gub)

                    #Zero Forces (Right Foot)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraints on Left Foot p_stance
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint on Left Foot p_stance
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)

            elif GaitPattern[Nph]=='DoubleSupport':
                #Get contact location
                #In the Double Support Phase, the p_stationary is the foot is the un-moved foot during StepNum (in second level)
                #the p_land is the landing/moving foot during StepNum (in the second level)
                if StepCnt == 0: #Special Case for the First Step (NOTE: Step 0)
                    p_stationary = ca.vertcat(px_init,py_init,pz_init)
                    p_stationary_TangentX = SurfTangentsX[0:3]
                    p_stationary_TangentY = SurfTangentsY[0:3]
                    p_stationary_Norm = SurfNorms[0:3]
                    p_stationary_Orientation = ca.reshape(SurfOriens[0:9],3,3).T

                    p_land = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                    p_land_TangentX = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_TangentY = SurfTangentsY[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_Norm = SurfNorms[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_Orientation = ca.reshape(SurfOriens[(StepCnt+1)*9:(StepCnt+1)*9+9],3,3).T
            
                else: #For other steps, indexed as 1,2,3,4
                    #In the Double Support Phase, the p_stationary is the foot is the un-moved foot during StepNum (in second level), \
                    #which is StepCnt - 1 (the moving foot in the previous step)
                    #the p_land is the landing/moving foot during StepNum (in the second level) --- StepCnt
                    #The StepCnt + 1 is the StepNum in the entire horizon (for getting terrain tangents and norm)
                    p_stationary = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                    p_stationary_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                    p_stationary_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                    p_stationary_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]
                    p_stationary_Orientation = ca.reshape(SurfOriens[StepCnt*9:StepCnt*9+9],3,3).T

                    p_land = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                    p_land_TangentX = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_TangentY = SurfTangentsY[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_Norm = SurfNorms[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_Orientation = ca.reshape(SurfOriens[(StepCnt+1)*9:(StepCnt+1)*9+9],3,3).T

                #Give Constraint according to even and odd steps
                if StepCnt%2 == 0: #Even Number of Steps
                    #-----------
                    #Case 1
                    #If First Level Swing the Left, then the second level Even Steps Swing the Right
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Left Foot is stationary
                    #Right Foot is the Land

                    #Kinemactics Constraint
                    #CoM in the Left (p_stationary)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_stationary, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                                 ContactFrameOrientation = p_stationary_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    #CoM in the Right (p_land)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_land, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                                 ContactFrameOrientation = p_land_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    
                    #   CoM Height Constraint (Left p_stationary foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = p_stationary_Orientation,
                                                           g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_land foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = p_land_Orientation,
                                                           g = g, glb = glb, gub = gub)

                    #Angular Dynamics
                    if AngularDynamics == True:
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator = SwingLegFlag, 
                                                                              Ldot_k = Ldot_k, CoM_k = CoM_k,
                                                                              PL = p_stationary, PL_TangentX = p_stationary_TangentX, PL_TangentY = p_stationary_TangentY, 
                                                                              PR = p_land,       PR_TangentX = p_land_TangentX,       PR_TangentY = p_land_TangentY, 
                                                                              FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, 
                                                                              FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k, g = g, glb = glb, gub = gub)

                    #Unilateral Constraint
                    #Left foot obey the unilateral constraint on p_stationary
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the Stationary foot p_land
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #Left foot obey the friction cone constraint on p_stationary
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the on p_land
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    
                    #------------
                    #Case 2
                    #If First Level Swing the Right, then the second level Even Steps Swing the Left
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Right Foot is stationary
                    #Left Foot is the Land

                    #Kinemactics Constraint
                    #CoM in the Left (p_land)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_land, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                                 ContactFrameOrientation = p_land_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    #CoM in the Right (p_stationary)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_stationary, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                                 ContactFrameOrientation = p_stationary_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    
                    #   CoM Height Constraint (Left p_land foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = p_land_Orientation,
                                                           g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_stationary foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = p_stationary_Orientation,
                                                           g = g, glb = glb, gub = gub)

                    #Angular Dynamics
                    if AngularDynamics == True:
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator = SwingLegFlag, 
                                                                              Ldot_k = Ldot_k, CoM_k = CoM_k,
                                                                              PL = p_land, PL_TangentX = p_land_TangentX, PL_TangentY = p_land_TangentY, 
                                                                              PR = p_stationary, PR_TangentX = p_stationary_TangentX, PR_TangentY = p_stationary_TangentY,  
                                                                              FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, 
                                                                              FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k, g = g, glb = glb, gub = gub)

                    #Unilateral Constraint
                    #Left foot obey the unilateral constraint on p_land
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the Stationary foot p_stationary
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #Left foot obey the friction cone constraint on p_land
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the on p_stationary
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    
                elif StepCnt%2 == 1:#Odd Number of Steps
                    #------
                    #Case 1
                    #If First Level Swing the Left, then the second level Odd Steps Swing the Left
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Right Foot is the Stationary
                    #Left Foot is the Land

                    #Kinemactics Constraint
                    #CoM in the Left (p_land)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_land, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                                 ContactFrameOrientation = p_land_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    #CoM in the Right (p_stationary)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_stationary, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                                 ContactFrameOrientation = p_stationary_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Left p_land foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = p_land_Orientation,
                                                           g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_stationary foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = p_stationary_Orientation,
                                                           g = g, glb = glb, gub = gub)

                    #Angular Dynamics
                    if AngularDynamics == True:
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator = SwingLegFlag, 
                                                                              Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                              PL = p_land, PL_TangentX = p_land_TangentX, PL_TangentY = p_land_TangentY, 
                                                                              PR = p_stationary, PR_TangentX = p_stationary_TangentX, PR_TangentY = p_stationary_TangentY, 
                                                                              FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, 
                                                                              FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k, g = g, glb = glb, gub = gub)

                    #Unilateral Constraint
                    #Left foot obey the unilateral constraint on p_land
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the Stationary foot p_stationary
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #Left foot obey the friction cone constraint on p_land
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the on p_stationary
                    g, glb, gub = FrictionCone(SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    
                
                    #------
                    #Case 2
                    #If First Level Swing the Right, then the second level Odd Steps Swing the Right
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Left Foot is the stationary
                    #Right Foot is the Land

                    #Kinematics Constraint
                    #CoM in the Left (p_stationary)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_stationary, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, 
                                                 ContactFrameOrientation = p_stationary_Orientation,
                                                 g = g, glb = glb, gub = gub)
                    #CoM in the Right (p_land)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                                 CoM_k = CoM_k, P = p_land, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, 
                                                 ContactFrameOrientation = p_land_Orientation, 
                                                 g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Left p_stationary foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = p_stationary_Orientation,
                                                           g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_land foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, 
                                                           ContactFrameOrientation = p_land_Orientation,
                                                           g = g, glb = glb, gub = gub)

                    #Angular Dynamics
                    if AngularDynamics == True:
                        if k<N_K-1:
                            g, glb, gub = Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator = SwingLegFlag, 
                                                                              Ldot_k = Ldot_k, CoM_k = CoM_k, 
                                                                              PL = p_stationary, PL_TangentX = p_stationary_TangentX, PL_TangentY = p_stationary_TangentY, 
                                                                              PR = p_land,       PR_TangentX = p_land_TangentX,       PR_TangentY = p_land_TangentY,
                                                                              FL1_k = FL1_k, FL2_k = FL2_k, FL3_k = FL3_k, FL4_k = FL4_k, 
                                                                              FR1_k = FR1_k, FR2_k = FR2_k, FR3_k = FR3_k, FR4_k = FR4_k, g = g, glb = glb, gub = gub)

                    #Unilateral Constraint
                    #Left foot obey the unilateral constraint on p_stationary
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the Stationary foot p_land
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #Left foot obey the friction cone constraint on p_stationary
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the on p_land
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)

                        #-------------------------------------
            
            else:
                raise Exception("Unknown Phase Name")
            
            # #-------------------------------------
            # #Dynamics Constraint
            if k < N_K - 1: #N_K - 1 the enumeration of the last knot, -1 the knot before the last knot
                #First-order Dynamics CoM x, y, z
                g, glb, gub = First_Order_Integrator(next_state = x[k+1], cur_state = x[k], cur_derivative = xdot[k], h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = y[k+1], cur_state = y[k], cur_derivative = ydot[k], h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = z[k+1], cur_state = z[k], cur_derivative = zdot[k], h = h, g = g, glb = glb, gub = gub)
                #First-order Dynamics CoMdot x, y, z
                g, glb, gub = First_Order_Integrator(next_state = xdot[k+1], cur_state = xdot[k], 
                                                     cur_derivative = FL1x[k]/m + FL2x[k]/m + FL3x[k]/m + FL4x[k]/m + FR1x[k]/m + FR2x[k]/m + FR3x[k]/m + FR4x[k]/m, 
                                                     h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = ydot[k+1], cur_state = ydot[k], 
                                                     cur_derivative = FL1y[k]/m + FL2y[k]/m + FL3y[k]/m + FL4y[k]/m + FR1y[k]/m + FR2y[k]/m + FR3y[k]/m + FR4y[k]/m, 
                                                     h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = zdot[k+1], cur_state = zdot[k], 
                                                     cur_derivative = FL1z[k]/m + FL2z[k]/m + FL3z[k]/m + FL4z[k]/m + FR1z[k]/m + FR2z[k]/m + FR3z[k]/m + FR4z[k]/m - G, 
                                                     h = h, g = g, glb = glb, gub = gub)
                #First-Order Dynamics L x, y, z
                g, glb, gub = First_Order_Integrator(next_state = Lx[k+1], cur_state = Lx[k], cur_derivative = Ldotx[k], h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = Ly[k+1], cur_state = Ly[k], cur_derivative = Ldoty[k], h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = Lz[k+1], cur_state = Lz[k], cur_derivative = Ldotz[k], h = h, g = g, glb = glb, gub = gub)
            
            #Cost Terms
            if k < N_K - 1:
                #with Acceleration
                J = J + h*(FL1x[k]/m + FL2x[k]/m + FL3x[k]/m + FL4x[k]/m + FR1x[k]/m + FR2x[k]/m + FR3x[k]/m + FR4x[k]/m)**2 + \
                        h*(FL1y[k]/m + FL2y[k]/m + FL3y[k]/m + FL4y[k]/m + FR1y[k]/m + FR2y[k]/m + FR3y[k]/m + FR4y[k]/m)**2 + \
                        h*(FL1z[k]/m + FL2z[k]/m + FL3z[k]/m + FL4z[k]/m + FR1z[k]/m + FR2z[k]/m + FR3z[k]/m + FR4z[k]/m - G)**2
                #with Angular Momentum
                J = J + h*Lx[k]**2 + h*Ly[k]**2 + h*Lz[k]**2
                #With Angular momentum rate
                #J = J + h*Ldotx[k]**2 + h*Ldoty[k]**2 + h*Ldotz[k]**2
    
    #-------------------------------------
    #Relative Footstep Constraint
    for step_cnt in range(Nsteps):
        if step_cnt == 0:
            #!!!!!!Pass from the first Level!!!!!!
            P_k_current = ca.vertcat(px_init,py_init,pz_init) #ca.vertcat(px[-1],py[-1],pz[-1])
            P_k_current_Orientation = ca.reshape(SurfOriens[0:9],3,3).T
            #!!!!!!
            P_k_next = ca.vertcat(px[step_cnt],py[step_cnt],pz[step_cnt])
            P_k_next_Orientation = ca.reshape(SurfOriens[(StepCnt+1)*9:(StepCnt+1)*9+9],3,3).T
        else:
            P_k_current = ca.vertcat(px[step_cnt-1],py[step_cnt-1],pz[step_cnt-1])
            P_k_current_Orientation = ca.reshape(SurfOriens[StepCnt*9:StepCnt*9+9],3,3).T

            P_k_next = ca.vertcat(px[step_cnt],py[step_cnt],pz[step_cnt])
            P_k_next_Orientation = ca.reshape(SurfOriens[(StepCnt+1)*9:(StepCnt+1)*9+9],3,3).T


        if step_cnt%2 == 0: #even number steps
            #----
            #Case 1
            #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right
            SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
            #Right Foot is landing (p_next), Left foot is stationary (p_current)
            #Relative Swing Foot Location - Landing (p_next/Right) in Stationary (p_current/Left) (rf in lf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                     p_next = P_k_next, p_cur = P_k_current, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, 
                                     ContactFrameOrientation = P_k_current_Orientation,
                                     g = g, glb = glb, gub = gub)
            #Relative Swing Foot Location - Stationary (p_current/Left) in Landing (p_next/Right) (lf in rf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                     p_next = P_k_current, p_cur = P_k_next, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, 
                                     ContactFrameOrientation = P_k_next_Orientation,
                                     g = g, glb = glb, gub = gub)

            #------
            #Case 2
            #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left
            SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
            #Left foot is landing (p_next), Right Foot is stationary (p_current)
            #Relative Swing Foot Location - Landing (p_next/Left) in stationary (p_current/Right) (lf in rf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                     p_next = P_k_next, p_cur = P_k_current, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, 
                                     ContactFrameOrientation = P_k_current_Orientation,
                                     g = g, glb = glb, gub = gub)
            #Relative Swing Foot Location - Stationary (p_current/Right) in Landing (p_next/Left) (rf in lf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                     p_next = P_k_current, p_cur = P_k_next, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, 
                                     ContactFrameOrientation = P_k_next_Orientation,
                                     g = g, glb = glb, gub = gub)

        elif step_cnt%2 == 1: #odd number steps
            #-------
            #Case 1
            #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left
            SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
            #Left Foot is landing (p_next), Right Foot is stationary (p_current)
            #Relative Swing Foot Location - Landing (p_next/Left) in stationary (p_current/Right) (lf in rf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                     p_next = P_k_next, p_cur = P_k_current, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, 
                                     ContactFrameOrientation = P_k_current_Orientation,
                                     g = g, glb = glb, gub = gub)
            #Relative Swing Foot Location - Stationary (p_current/Right) in Landing (p_next/Left) (rf in lf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                     p_next = P_k_current, p_cur = P_k_next, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, 
                                     ContactFrameOrientation = P_k_next_Orientation,
                                     g = g, glb = glb, gub = gub)

            #-------
            #Case 2
            #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right
            SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
            #Right Foot is landing (p_next), Left foot is stationary (p_current)
            #Relative Swing Foot Location - Landing (p_next/Right) in Stationary (p_current/Left) (rf in lf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                     p_next = P_k_next, p_cur = P_k_current, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, 
                                     ContactFrameOrientation = P_k_current_Orientation,
                                     g = g, glb = glb, gub = gub)
            #Relative Swing Foot Location - Stationary (p_current/Left) in Landing (p_next/Right) (lf in rf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, 
                                     p_next = P_k_current, p_cur = P_k_next, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, 
                                     ContactFrameOrientation = P_k_next_Orientation,
                                     g = g, glb = glb, gub = gub)

    #-----------------
    #FootStep Location Constraint (On the Patch) -> Only One Step
    #P3----------------P1
    #|                  |
    #|                  |
    #|                  |
    #P4----------------P2

    for PatchNum in range(Nsteps): #No need to consider p_init, as they are constrained by the first level
        #Get Footstep Vector
        P_vector = ca.vertcat(px[PatchNum],py[PatchNum],pz[PatchNum])

        #Get Half Space Representation 
        #NOTE: In the second level, the terrain patch start from the second patch, indexed as 1
        SurfParaTemp = SurfParas[20+PatchNum*20:19+(PatchNum+1)*20+1]
        #print(SurfParaTemp)
        SurfK = SurfParaTemp[0:11+1]
        SurfK = ca.reshape(SurfK,3,4)
        SurfK = SurfK.T #NOTE: This process is due to casadi naming convention to have first row to be x1,x2,x3
        SurfE = SurfParaTemp[11+1:11+3+1]
        Surfk = SurfParaTemp[14+1:14+4+1]
        Surfe = SurfParaTemp[-1]

        #Terrain Tangent and Norms
        #In the second level, when getting the tangets, we need to have PatchNum/StepCnt + 1, 
        #As the second level counts the second step (enumerated as step 1 in entire horizon) as step 0
        P_vector_TangentX = SurfTangentsX[(PatchNum+1)*3:(PatchNum+1)*3+3]
        P_vector_TangentY = SurfTangentsY[(PatchNum+1)*3:(PatchNum+1)*3+3]

        g, glb, gub = Stay_on_Surf(P = P_vector, P_TangentX = P_vector_TangentX, P_TangentY = P_vector_TangentY, 
                                   ineq_K = SurfK, ineq_k = Surfk, eq_E = SurfE, eq_e = Surfe, g = g, glb = glb, gub = gub)

    #-----------------------------------
    #Switching Time Constraint
    for phase_cnt in range(Nphase):
        if GaitPattern[phase_cnt] == 'InitialDouble':
            if phase_cnt == 0:
                g.append(Ts[phase_cnt] - 0)
                glb.append(np.array([PhaseDuration_Limits["DoubleSupport_Min"]])) #old 0.1-0.3
                gub.append(np.array([PhaseDuration_Limits["DoubleSupport_Max"]]))
            else:
                g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
                glb.append(np.array([PhaseDuration_Limits["DoubleSupport_Min"]]))
                gub.append(np.array([PhaseDuration_Limits["DoubleSupport_Max"]]))
        elif GaitPattern[phase_cnt] == 'Swing':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
            glb.append(np.array([PhaseDuration_Limits["SingleSupport_Min"]])) #0.8-1.2
            gub.append(np.array([PhaseDuration_Limits["SingleSupport_Max"]]))
        elif GaitPattern[phase_cnt] == 'DoubleSupport':
            g.append(Ts[phase_cnt]-Ts[phase_cnt-1]) #0.1-0.9
            glb.append(np.array([PhaseDuration_Limits["DoubleSupport_Min"]]))
            gub.append(np.array([PhaseDuration_Limits["DoubleSupport_Max"]])) #old - 0.1-0.3
        else:
            raise Exception("Unknown Phase Name")

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Variable Index - !!!This is the pure Index, when try to get the array using other routines, we need to add "+1" at the last index due to Python indexing conventions
    #First set of variables start counting from 0, The enumeration of the last knot is N_K-1
    x_index = (0,N_K-1);                                      y_index = (x_index[1]+1,x_index[1]+N_K);                 z_index = (y_index[1]+1,y_index[1]+N_K)
    xdot_index = (z_index[1]+1,z_index[1]+N_K);               ydot_index = (xdot_index[1]+1,xdot_index[1]+N_K);        zdot_index = (ydot_index[1]+1,ydot_index[1]+N_K)
    Lx_index = (zdot_index[1]+1,zdot_index[1]+N_K);           Ly_index = (Lx_index[1]+1,Lx_index[1]+N_K);              Lz_index = (Ly_index[1]+1,Ly_index[1]+N_K)
    Ldotx_index = (Lz_index[1]+1,Lz_index[1]+N_K);            Ldoty_index = (Ldotx_index[1]+1,Ldotx_index[1]+N_K);     Ldotz_index = (Ldoty_index[1]+1,Ldoty_index[1]+N_K)
    FL1x_index = (Ldotz_index[1]+1,Ldotz_index[1]+N_K);       FL1y_index = (FL1x_index[1]+1,FL1x_index[1]+N_K);        FL1z_index = (FL1y_index[1]+1,FL1y_index[1]+N_K)
    FL2x_index = (FL1z_index[1]+1,FL1z_index[1]+N_K);         FL2y_index = (FL2x_index[1]+1,FL2x_index[1]+N_K);        FL2z_index = (FL2y_index[1]+1,FL2y_index[1]+N_K)
    FL3x_index = (FL2z_index[1]+1,FL2z_index[1]+N_K);         FL3y_index = (FL3x_index[1]+1,FL3x_index[1]+N_K);        FL3z_index = (FL3y_index[1]+1,FL3y_index[1]+N_K)
    FL4x_index = (FL3z_index[1]+1,FL3z_index[1]+N_K);         FL4y_index = (FL4x_index[1]+1,FL4x_index[1]+N_K);        FL4z_index = (FL4y_index[1]+1,FL4y_index[1]+N_K)
    FR1x_index = (FL4z_index[1]+1,FL4z_index[1]+N_K);         FR1y_index = (FR1x_index[1]+1,FR1x_index[1]+N_K);        FR1z_index = (FR1y_index[1]+1,FR1y_index[1]+N_K)
    FR2x_index = (FR1z_index[1]+1,FR1z_index[1]+N_K);         FR2y_index = (FR2x_index[1]+1,FR2x_index[1]+N_K);        FR2z_index = (FR2y_index[1]+1,FR2y_index[1]+N_K)
    FR3x_index = (FR2z_index[1]+1,FR2z_index[1]+N_K);         FR3y_index = (FR3x_index[1]+1,FR3x_index[1]+N_K);        FR3z_index = (FR3y_index[1]+1,FR3y_index[1]+N_K)
    FR4x_index = (FR3z_index[1]+1,FR3z_index[1]+N_K);         FR4y_index = (FR4x_index[1]+1,FR4x_index[1]+N_K);        FR4z_index = (FR4y_index[1]+1,FR4y_index[1]+N_K)
    px_init_index = (FR4z_index[1]+1,FR4z_index[1]+1);        py_init_index = (px_init_index[1]+1,px_init_index[1]+1); pz_init_index = (py_init_index[1]+1,py_init_index[1]+1)
    px_index = (pz_init_index[1]+1,pz_init_index[1]+Nsteps);  py_index = (px_index[1]+1,px_index[1]+Nsteps);           pz_index = (py_index[1]+1,py_index[1]+Nsteps)
    Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)

    var_index = {"x":x_index,        "y":y_index,         "z":z_index,         "xdot":xdot_index,     "ydot":ydot_index,     "zdot":zdot_index,
                 "Lx":Lx_index,      "Ly":Ly_index,       "Lz":Lz_index,       "Ldotx":Ldotx_index,   "Ldoty":Ldoty_index,   "Ldotz":Ldotz_index,
                 "FL1x":FL1x_index,  "FL1y":FL1y_index,   "FL1z":FL1z_index,   
                 "FL2x":FL2x_index,  "FL2y":FL2y_index,   "FL2z":FL2z_index,
                 "FL3x":FL3x_index,  "FL3y":FL3y_index,   "FL3z":FL3z_index,
                 "FL4x":FL4x_index,  "FL4y":FL4y_index,   "FL4z":FL4z_index,
                 "FR1x":FR1x_index,  "FR1y":FR1y_index,   "FR1z":FR1z_index,
                 "FR2x":FR2x_index,  "FR2y":FR2y_index,   "FR2z":FR2z_index,
                 "FR3x":FR3x_index,  "FR3y":FR3y_index,   "FR3z":FR3z_index,
                 "FR4x":FR4x_index,  "FR4y":FR4y_index,   "FR4z":FR4z_index,
                 "px_init":px_init_index,   "py_init":py_init_index,   "pz_init":pz_init_index,
                 "px":px_index,   "py":py_index,   "pz":pz_index,
                 "Ts":Ts_index,
    }
    return DecisionVars, DecisionVars_lb, DecisionVars_ub, J, g, glb, gub, var_index


#NOTE: Ponton Methods do not have rotated kinematics polytopes
def Ponton_FourPoints(m = 100.0, Nk_Local = 7, Nsteps = 1, ParameterList = None, PontonTerm_bounds = 0.55):
    #-------------------------------------------
    #Define Constant Parameters
    #   Gravitational Acceleration
    G = 9.80665 #kg/m^2
    #   Friciton Coefficient 
    miu = 0.3
    #   Force Limits
    F_bound = 400.0
    Fxlb = -F_bound;   Fxub = F_bound
    Fylb = -F_bound;   Fyub = F_bound
    Fzlb = -F_bound;   Fzub = F_bound
    #   Angular Momentum Bounds
    #L_bound = 2.5;      Llb = -L_bound;         Lub = L_bound
    #Ldot_bound = 3.5;   Ldotlb = -Ldot_bound;   Ldotub = Ldot_bound
    #   Bounds on CoM Height
    z_lowest = -5.0
    z_highest = 5.0
    #   CoM Height with respect to Footstep Location
    CoM_z_to_Foot_min = 0.65 #0.6
    CoM_z_to_Foot_max = 0.75
    #   Ponton Term Bounds
    p_lb = -PontonTerm_bounds;     p_ub = PontonTerm_bounds
    q_lb = -PontonTerm_bounds;     q_ub = PontonTerm_bounds
    #   Leg Length (Normalisation) for Ponton
    max_leg_length = 1.5 #can be 1.45
    #---------------------------------------------
    #Define Parameters
    #   Gait Pattern, Each action is followed up by a double support phase
    GaitPattern = ["InitialDouble","Swing","DoubleSupport"] + ["InitialDouble", "Swing","DoubleSupport"]*(Nsteps-1) #,'RightSupport','DoubleSupport','LeftSupport','DoubleSupport'
    #   Number of Phases
    Nphase = len(GaitPattern)
    #   Compute Number of Total knots/ticks, but the enumeration start from 0 to N_K-1
    N_K = Nk_Local*Nphase + 1 #+1 the last knots to finalize the plan
    #   Phase Duration Vector; NOTE: Mannually defined
    #PhaseDurationVec = [0.3, 0.8, 0.3]*(Nsteps) + [0.3, 0.8, 0.3]*(Nsteps-1)
    PhaseDurationVec = [0.2, 0.5, 0.2]*(Nsteps) + [0.2, 0.5, 0.2]*(Nsteps-1)

    #-----------------------------------------------------------------------------------------------------------------------
    # #Load kinematics Polytope
    # #   Not local than server
    # kinefilepath = "/home/jiayu/Desktop/multicontact_learning_local_objectives/python/kinematics_polytope/kinematics_constraints.p"
    # if os.path.isfile(kinefilepath) == False:
    #     kinefilepath = "/afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/kinematics_polytope/kinematics_constraints.p"

    # with open(kinefilepath, 'rb') as f:
    #     kinematics_constraints= pickle.load(f)
    
    # #CoM Polytopes
    # K_CoM_Right = kinematics_constraints["K_CoM_in_Right_Contact"];     k_CoM_Right = kinematics_constraints["k_CoM_in_Right_Contact"]
    # K_CoM_Left  = kinematics_constraints["K_CoM_in_Left_Contact"];      k_CoM_Left  = kinematics_constraints["k_CoM_in_Left_Contact"]

    # #Relative Footstep constraints
    # Q_rf_in_lf = kinematics_constraints["Q_Right_Contact_in_Left_Contact"];    q_rf_in_lf = kinematics_constraints["q_Right_Contact_in_Left_Contact"]
    # Q_lf_in_rf = kinematics_constraints["Q_Left_Contact_in_Right_Contact"];    q_lf_in_rf = kinematics_constraints["q_Left_Contact_in_Right_Contact"]

    #Get Kinematics Constraint for Talos
    #CoM kinematics constraint, give homogenous transformaiton (the last column seems like dont make a diff)
    K_CoM_Right,k_CoM_Right = right_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    K_CoM_Left,k_CoM_Left = left_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    #Another way
    #kinematicConstraints = genKinematicConstraints(left_foot_constraints,right_foot_constraints)
    #K_CoM_Left = kinematicConstraints[0][0];   k_CoM_Left = kinematicConstraints[0][1]
    #K_CoM_Right = kinematicConstraints[1][0];  k_CoM_Right = kinematicConstraints[1][1]
    
    #Relative Foot Constraint matrices
    #Relative foot constraint, give homogenous transformation (the last column seems like dont make a diff)
    Q_rf_in_lf,q_rf_in_lf = right_foot_in_lf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    Q_lf_in_rf,q_lf_in_rf = left_foot_in_rf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    #Another way
    #relativeConstraints = genFootRelativeConstraints(right_foot_in_lf_frame_constraints,left_foot_in_rf_frame_constraints)
    #Q_rf_in_lf = relativeConstraints[0][0];   q_rf_in_lf = relativeConstraints[0][1] #named lf in rf, but representing rf in lf
    #Q_lf_in_rf = relativeConstraints[1][0];   q_lf_in_rf = relativeConstraints[1][1] #named rf in lf, but representing lf in rf

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Casadi Parameters
    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ParameterList["LeftSwingFlag"]
    ParaRightSwingFlag = ParameterList["RightSwingFlag"]

    #Initial Left Foot Contact Location
    PLx_init = ParameterList["PLx_init"];   PLy_init = ParameterList["PLy_init"];   PLz_init = ParameterList["PLz_init"]
    PL_init = ca.vertcat(PLx_init,PLy_init,PLz_init)
    #Initial Right Foot Contact Location
    PRx_init = ParameterList["PRx_init"];   PRy_init = ParameterList["PRy_init"];   PRz_init = ParameterList["PRz_init"]
    PR_init = ca.vertcat(PRx_init,PRy_init,PRz_init)

    #Surface Patches
    SurfParas = ParameterList["SurfParas"]

    #Tangents and Norms
    #Initial Contact Norm and Tangents
    PL_init_Norm = ParameterList["PL_init_Norm"];   PL_init_TangentX = ParameterList["PL_init_TangentX"];   PL_init_TangentY = ParameterList["PL_init_TangentY"]
    PR_init_Norm = ParameterList["PR_init_Norm"];   PR_init_TangentX = ParameterList["PR_init_TangentX"];   PR_init_TangentY = ParameterList["PR_init_TangentY"]

    #Future Contact Norm and Tangents
    SurfNorms = ParameterList["SurfNorms"]                
    SurfTangentsX = ParameterList["SurfTangentsX"]
    SurfTangentsY = ParameterList["SurfTangentsY"]

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Variables and Bounds, Parameters
    #   CoM Position x, y, z
    x = ca.SX.sym('x',N_K);   x_lb = np.array([[0.0]*(x.shape[0]*x.shape[1])]);         x_ub = np.array([[50.0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    y = ca.SX.sym('y',N_K);   y_lb = np.array([[-1.0]*(y.shape[0]*y.shape[1])]);        y_ub = np.array([[1.0]*(y.shape[0]*y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    z = ca.SX.sym('z',N_K);   z_lb = np.array([[z_lowest]*(z.shape[0]*z.shape[1])]);  z_ub = np.array([[z_highest]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #   CoM Velocity x, y, z
    xdot = ca.SX.sym('xdot',N_K);   xdot_lb = np.array([[-1.5]*(xdot.shape[0]*xdot.shape[1])]);   xdot_ub = np.array([[1.5]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    ydot = ca.SX.sym('ydot',N_K);   ydot_lb = np.array([[-1.5]*(ydot.shape[0]*ydot.shape[1])]);   ydot_ub = np.array([[1.5]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    zdot = ca.SX.sym('zdot',N_K);   zdot_lb = np.array([[-1.5]*(zdot.shape[0]*zdot.shape[1])]);   zdot_ub = np.array([[1.5]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Left Foot Forces
    #Left Foot Contact Point 1 x, y, z
    FL1x = ca.SX.sym('FL1x',N_K);   FL1x_lb = np.array([[Fxlb]*(FL1x.shape[0]*FL1x.shape[1])]);   FL1x_ub = np.array([[Fxub]*(FL1x.shape[0]*FL1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1y = ca.SX.sym('FL1y',N_K);   FL1y_lb = np.array([[Fylb]*(FL1y.shape[0]*FL1y.shape[1])]);   FL1y_ub = np.array([[Fyub]*(FL1y.shape[0]*FL1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL1z = ca.SX.sym('FL1z',N_K);   FL1z_lb = np.array([[Fzlb]*(FL1z.shape[0]*FL1z.shape[1])]);   FL1z_ub = np.array([[Fzub]*(FL1z.shape[0]*FL1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Left Foot Contact Point 2 x, y, z
    FL2x = ca.SX.sym('FL2x',N_K);   FL2x_lb = np.array([[Fxlb]*(FL2x.shape[0]*FL2x.shape[1])]);   FL2x_ub = np.array([[Fxub]*(FL2x.shape[0]*FL2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2y = ca.SX.sym('FL2y',N_K);   FL2y_lb = np.array([[Fylb]*(FL2y.shape[0]*FL2y.shape[1])]);   FL2y_ub = np.array([[Fyub]*(FL2y.shape[0]*FL2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL2z = ca.SX.sym('FL2z',N_K);   FL2z_lb = np.array([[Fzlb]*(FL2z.shape[0]*FL2z.shape[1])]);   FL2z_ub = np.array([[Fzub]*(FL2z.shape[0]*FL2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Left Foot Contact Point 3 x, y, z
    FL3x = ca.SX.sym('FL3x',N_K);   FL3x_lb = np.array([[Fxlb]*(FL3x.shape[0]*FL3x.shape[1])]);   FL3x_ub = np.array([[Fxub]*(FL3x.shape[0]*FL3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3y = ca.SX.sym('FL3y',N_K);   FL3y_lb = np.array([[Fylb]*(FL3y.shape[0]*FL3y.shape[1])]);   FL3y_ub = np.array([[Fyub]*(FL3y.shape[0]*FL3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL3z = ca.SX.sym('FL3z',N_K);   FL3z_lb = np.array([[Fzlb]*(FL3z.shape[0]*FL3z.shape[1])]);   FL3z_ub = np.array([[Fzub]*(FL3z.shape[0]*FL3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Left Foot Contact Point 4 x-axis
    FL4x = ca.SX.sym('FL4x',N_K);   FL4x_lb = np.array([[Fxlb]*(FL4x.shape[0]*FL4x.shape[1])]);   FL4x_ub = np.array([[Fxub]*(FL4x.shape[0]*FL4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4y = ca.SX.sym('FL4y',N_K);   FL4y_lb = np.array([[Fylb]*(FL4y.shape[0]*FL4y.shape[1])]);   FL4y_ub = np.array([[Fyub]*(FL4y.shape[0]*FL4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FL4z = ca.SX.sym('FL4z',N_K);   FL4z_lb = np.array([[Fzlb]*(FL4z.shape[0]*FL4z.shape[1])]);   FL4z_ub = np.array([[Fzub]*(FL4z.shape[0]*FL4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Right Contact Force x-axis
    #Right Foot Contact Point 1 x, y, z
    FR1x = ca.SX.sym('FR1x',N_K);   FR1x_lb = np.array([[Fxlb]*(FR1x.shape[0]*FR1x.shape[1])]);   FR1x_ub = np.array([[Fxub]*(FR1x.shape[0]*FR1x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1y = ca.SX.sym('FR1y',N_K);   FR1y_lb = np.array([[Fylb]*(FR1y.shape[0]*FR1y.shape[1])]);   FR1y_ub = np.array([[Fyub]*(FR1y.shape[0]*FR1y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR1z = ca.SX.sym('FR1z',N_K);   FR1z_lb = np.array([[Fzlb]*(FR1z.shape[0]*FR1z.shape[1])]);   FR1z_ub = np.array([[Fzub]*(FR1z.shape[0]*FR1z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Right Foot Contact Point 2 x, y, z
    FR2x = ca.SX.sym('FR2x',N_K);   FR2x_lb = np.array([[Fxlb]*(FR2x.shape[0]*FR2x.shape[1])]);   FR2x_ub = np.array([[Fxub]*(FR2x.shape[0]*FR2x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2y = ca.SX.sym('FR2y',N_K);   FR2y_lb = np.array([[Fylb]*(FR2y.shape[0]*FR2y.shape[1])]);   FR2y_ub = np.array([[Fyub]*(FR2y.shape[0]*FR2y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR2z = ca.SX.sym('FR2z',N_K);   FR2z_lb = np.array([[Fzlb]*(FR2z.shape[0]*FR2z.shape[1])]);   FR2z_ub = np.array([[Fzub]*(FR2z.shape[0]*FR2z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Right Foot Contact Point 3 x, y, z
    FR3x = ca.SX.sym('FR3x',N_K);   FR3x_lb = np.array([[Fxlb]*(FR3x.shape[0]*FR3x.shape[1])]);   FR3x_ub = np.array([[Fxub]*(FR3x.shape[0]*FR3x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3y = ca.SX.sym('FR3y',N_K);   FR3y_lb = np.array([[Fylb]*(FR3y.shape[0]*FR3y.shape[1])]);   FR3y_ub = np.array([[Fyub]*(FR3y.shape[0]*FR3y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR3z = ca.SX.sym('FR3z',N_K);   FR3z_lb = np.array([[Fzlb]*(FR3z.shape[0]*FR3z.shape[1])]);   FR3z_ub = np.array([[Fzub]*(FR3z.shape[0]*FR3z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Right Foot Contact Point 4 x, y, z
    FR4x = ca.SX.sym('FR4x',N_K);   FR4x_lb = np.array([[Fxlb]*(FR4x.shape[0]*FR4x.shape[1])]);   FR4x_ub = np.array([[Fxub]*(FR4x.shape[0]*FR4x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4y = ca.SX.sym('FR4y',N_K);   FR4y_lb = np.array([[Fylb]*(FR4y.shape[0]*FR4y.shape[1])]);   FR4y_ub = np.array([[Fyub]*(FR4y.shape[0]*FR4y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FR4z = ca.SX.sym('FR4z',N_K);   FR4z_lb = np.array([[Fzlb]*(FR4z.shape[0]*FR4z.shape[1])]);   FR4z_ub = np.array([[Fzub]*(FR4z.shape[0]*FR4z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    
    #Initial Contact Location (First step made in the first level), need to connect to the first level landing location 
    #   Px, Py, Pz
    px_init = ca.SX.sym('px_init');   px_init_lb = np.array([-1.0]);   px_init_ub = np.array([50.0])
    py_init = ca.SX.sym('py_init');   py_init_lb = np.array([-2.0]);   py_init_ub = np.array([2.0])
    pz_init = ca.SX.sym('pz_init');   pz_init_lb = np.array([-5.0]);   pz_init_ub = np.array([5.0])
    
    #Contact Location Sequence
    px = [];   px_lb = [];   px_ub = []
    py = [];   py_lb = [];   py_ub = []
    pz = [];   pz_lb = [];   pz_ub = []
    for stepIdx in range(Nsteps):
        #Nsteps: Number of steps in the second level, = Total Number of Steps of the Entire Lookahead Horizon - 1
        #Therefore the enumeration of contact location sequences start counting from 1 (to be aligned with step number in the entire horizon)
        pxtemp = ca.SX.sym('px'+str(stepIdx + 1));   px.append(pxtemp);   px_lb.append(np.array([-1.0]));   px_ub.append(np.array([50.0]))
        pytemp = ca.SX.sym('py'+str(stepIdx + 1));   py.append(pytemp);   py_lb.append(np.array([-2.0]));   py_ub.append(np.array([2.0]))
        pztemp = ca.SX.sym('pz'+str(stepIdx + 1));   pz.append(pztemp);   pz_lb.append(np.array([-5.0]));   pz_ub.append(np.array([5.0]))

    #Switching Time Vector
    Ts = [];   Ts_lb = [];   Ts_ub = []
    for n_phase in range(Nphase):
        #Ts start counting from 1, Ts0 = 0
        Tstemp = ca.SX.sym('Ts'+str(n_phase+1));   Ts.append(Tstemp);   Ts_lb.append(np.array([0.05]));   Ts_ub.append(np.array([3.0*(Nphase+1)]))    
    
    #Ponton Terms (p and q for x, y, z angular momentum rate)
    #Left Foot Contact Point 1 (p and q for x, y, z)
    #   x-axis
    cL1x_p = ca.SX.sym('cL1x_p',N_K);  cL1x_p_lb = np.array([[p_lb]*(cL1x_p.shape[0]*cL1x_p.shape[1])]);  cL1x_p_ub = np.array([[p_ub]*(cL1x_p.shape[0]*cL1x_p.shape[1])])
    cL1x_q = ca.SX.sym('cL1x_q',N_K);  cL1x_q_lb = np.array([[q_lb]*(cL1x_q.shape[0]*cL1x_q.shape[1])]);  cL1x_q_ub = np.array([[q_ub]*(cL1x_q.shape[0]*cL1x_q.shape[1])])
    #   y-axis
    cL1y_p = ca.SX.sym('cL1y_p',N_K);  cL1y_p_lb = np.array([[p_lb]*(cL1y_p.shape[0]*cL1y_p.shape[1])]);  cL1y_p_ub = np.array([[p_ub]*(cL1y_p.shape[0]*cL1y_p.shape[1])])
    cL1y_q = ca.SX.sym('cL1y_q',N_K);  cL1y_q_lb = np.array([[q_lb]*(cL1y_q.shape[0]*cL1y_q.shape[1])]);  cL1y_q_ub = np.array([[q_ub]*(cL1y_q.shape[0]*cL1y_q.shape[1])])
    #   z-axis
    cL1z_p = ca.SX.sym('cL1z_p',N_K);  cL1z_p_lb = np.array([[p_lb]*(cL1z_p.shape[0]*cL1z_p.shape[1])]);  cL1z_p_ub = np.array([[p_ub]*(cL1z_p.shape[0]*cL1z_p.shape[1])])
    cL1z_q = ca.SX.sym('cL1z_q',N_K);  cL1z_q_lb = np.array([[q_lb]*(cL1z_q.shape[0]*cL1z_q.shape[1])]);  cL1z_q_ub = np.array([[q_ub]*(cL1z_q.shape[0]*cL1z_q.shape[1])])
    #Left Foot Contact Point 2 p and q for x, y, z
    #   x-axis
    cL2x_p = ca.SX.sym('cL2x_p',N_K);  cL2x_p_lb = np.array([[p_lb]*(cL2x_p.shape[0]*cL2x_p.shape[1])]);  cL2x_p_ub = np.array([[p_ub]*(cL2x_p.shape[0]*cL2x_p.shape[1])])
    cL2x_q = ca.SX.sym('cL2x_q',N_K);  cL2x_q_lb = np.array([[q_lb]*(cL2x_q.shape[0]*cL2x_q.shape[1])]);  cL2x_q_ub = np.array([[q_ub]*(cL2x_q.shape[0]*cL2x_q.shape[1])])
    #   y-axis
    cL2y_p = ca.SX.sym('cL2y_p',N_K);  cL2y_p_lb = np.array([[p_lb]*(cL2y_p.shape[0]*cL2y_p.shape[1])]);  cL2y_p_ub = np.array([[p_ub]*(cL2y_p.shape[0]*cL2y_p.shape[1])])
    cL2y_q = ca.SX.sym('cL2y_q',N_K);  cL2y_q_lb = np.array([[q_lb]*(cL2y_q.shape[0]*cL2y_q.shape[1])]);  cL2y_q_ub = np.array([[q_ub]*(cL2y_q.shape[0]*cL2y_q.shape[1])])
    #   z-axis
    cL2z_p = ca.SX.sym('cL2z_p',N_K);  cL2z_p_lb = np.array([[p_lb]*(cL2z_p.shape[0]*cL2z_p.shape[1])]);  cL2z_p_ub = np.array([[p_ub]*(cL2z_p.shape[0]*cL2z_p.shape[1])])
    cL2z_q = ca.SX.sym('cL2z_q',N_K);  cL2z_q_lb = np.array([[q_lb]*(cL2z_q.shape[0]*cL2z_q.shape[1])]);  cL2z_q_ub = np.array([[q_ub]*(cL2z_q.shape[0]*cL2z_q.shape[1])])
    #Left Foot Contact Point 3 p and q for x, y, z
    #   x-axis
    cL3x_p = ca.SX.sym('cL3x_p',N_K);  cL3x_p_lb = np.array([[p_lb]*(cL3x_p.shape[0]*cL3x_p.shape[1])]);  cL3x_p_ub = np.array([[p_ub]*(cL3x_p.shape[0]*cL3x_p.shape[1])])
    cL3x_q = ca.SX.sym('cL3x_q',N_K);  cL3x_q_lb = np.array([[q_lb]*(cL3x_q.shape[0]*cL3x_q.shape[1])]);  cL3x_q_ub = np.array([[q_ub]*(cL3x_q.shape[0]*cL3x_q.shape[1])])
    #   y-axis
    cL3y_p = ca.SX.sym('cL3y_p',N_K);  cL3y_p_lb = np.array([[p_lb]*(cL3y_p.shape[0]*cL3y_p.shape[1])]);  cL3y_p_ub = np.array([[p_ub]*(cL3y_p.shape[0]*cL3y_p.shape[1])])
    cL3y_q = ca.SX.sym('cL3y_q',N_K);  cL3y_q_lb = np.array([[q_lb]*(cL3y_q.shape[0]*cL3y_q.shape[1])]);  cL3y_q_ub = np.array([[q_ub]*(cL3y_q.shape[0]*cL3y_q.shape[1])])
    #   z-axis
    cL3z_p = ca.SX.sym('cL3z_p',N_K);  cL3z_p_lb = np.array([[p_lb]*(cL3z_p.shape[0]*cL3z_p.shape[1])]);  cL3z_p_ub = np.array([[p_ub]*(cL3z_p.shape[0]*cL3z_p.shape[1])])
    cL3z_q = ca.SX.sym('cL3z_q',N_K);  cL3z_q_lb = np.array([[q_lb]*(cL3z_q.shape[0]*cL3z_q.shape[1])]);  cL3z_q_ub = np.array([[q_ub]*(cL3z_q.shape[0]*cL3z_q.shape[1])])
    #Left Foot Contact Point 4 p and q for x, y, z
    #   x-axis
    cL4x_p = ca.SX.sym('cL4x_p',N_K);  cL4x_p_lb = np.array([[p_lb]*(cL4x_p.shape[0]*cL4x_p.shape[1])]);  cL4x_p_ub = np.array([[p_ub]*(cL4x_p.shape[0]*cL4x_p.shape[1])])
    cL4x_q = ca.SX.sym('cL4x_q',N_K);  cL4x_q_lb = np.array([[q_lb]*(cL4x_q.shape[0]*cL4x_q.shape[1])]);  cL4x_q_ub = np.array([[q_ub]*(cL4x_q.shape[0]*cL4x_q.shape[1])])
    #   y-axis
    cL4y_p = ca.SX.sym('cL4y_p',N_K);  cL4y_p_lb = np.array([[p_lb]*(cL4y_p.shape[0]*cL4y_p.shape[1])]);  cL4y_p_ub = np.array([[p_ub]*(cL4y_p.shape[0]*cL4y_p.shape[1])])
    cL4y_q = ca.SX.sym('cL4y_q',N_K);  cL4y_q_lb = np.array([[q_lb]*(cL4y_q.shape[0]*cL4y_q.shape[1])]);  cL4y_q_ub = np.array([[q_ub]*(cL4y_q.shape[0]*cL4y_q.shape[1])])
    #   z-axis
    cL4z_p = ca.SX.sym('cL4z_p',N_K);  cL4z_p_lb = np.array([[p_lb]*(cL4z_p.shape[0]*cL4z_p.shape[1])]);  cL4z_p_ub = np.array([[p_ub]*(cL4z_p.shape[0]*cL4z_p.shape[1])])
    cL4z_q = ca.SX.sym('cL4z_q',N_K);  cL4z_q_lb = np.array([[q_lb]*(cL4z_q.shape[0]*cL4z_q.shape[1])]);  cL4z_q_ub = np.array([[q_ub]*(cL4z_q.shape[0]*cL4z_q.shape[1])])    
    #Right Foot Contact Point 1 p and q for x, y, z
    #   x-axis
    cR1x_p = ca.SX.sym('cR1x_p',N_K);  cR1x_p_lb = np.array([[p_lb]*(cR1x_p.shape[0]*cR1x_p.shape[1])]);  cR1x_p_ub = np.array([[p_ub]*(cR1x_p.shape[0]*cR1x_p.shape[1])])
    cR1x_q = ca.SX.sym('cR1x_q',N_K);  cR1x_q_lb = np.array([[q_lb]*(cR1x_q.shape[0]*cR1x_q.shape[1])]);  cR1x_q_ub = np.array([[q_ub]*(cR1x_q.shape[0]*cR1x_q.shape[1])])
    #   y-axis
    cR1y_p = ca.SX.sym('cR1y_p',N_K);  cR1y_p_lb = np.array([[p_lb]*(cR1y_p.shape[0]*cR1y_p.shape[1])]);  cR1y_p_ub = np.array([[p_ub]*(cR1y_p.shape[0]*cR1y_p.shape[1])])
    cR1y_q = ca.SX.sym('cR1y_q',N_K);  cR1y_q_lb = np.array([[q_lb]*(cR1y_q.shape[0]*cR1y_q.shape[1])]);  cR1y_q_ub = np.array([[q_ub]*(cR1y_q.shape[0]*cR1y_q.shape[1])])
    #   z-axis
    cR1z_p = ca.SX.sym('cR1z_p',N_K);  cR1z_p_lb = np.array([[p_lb]*(cR1z_p.shape[0]*cR1z_p.shape[1])]);  cR1z_p_ub = np.array([[p_ub]*(cR1z_p.shape[0]*cR1z_p.shape[1])])
    cR1z_q = ca.SX.sym('cR1z_q',N_K);  cR1z_q_lb = np.array([[q_lb]*(cR1z_q.shape[0]*cR1z_q.shape[1])]);  cR1z_q_ub = np.array([[q_ub]*(cR1z_q.shape[0]*cR1z_q.shape[1])])    
    #Right Foot Contact Point 2 p and q for x, y, z
    #   x-axis
    cR2x_p = ca.SX.sym('cR2x_p',N_K);  cR2x_p_lb = np.array([[p_lb]*(cR2x_p.shape[0]*cR2x_p.shape[1])]);  cR2x_p_ub = np.array([[p_ub]*(cR2x_p.shape[0]*cR2x_p.shape[1])])
    cR2x_q = ca.SX.sym('cR2x_q',N_K);  cR2x_q_lb = np.array([[q_lb]*(cR2x_q.shape[0]*cR2x_q.shape[1])]);  cR2x_q_ub = np.array([[q_ub]*(cR2x_q.shape[0]*cR2x_q.shape[1])])
    #   y-axis
    cR2y_p = ca.SX.sym('cR2y_p',N_K);  cR2y_p_lb = np.array([[p_lb]*(cR2y_p.shape[0]*cR2y_p.shape[1])]);  cR2y_p_ub = np.array([[p_ub]*(cR2y_p.shape[0]*cR2y_p.shape[1])])
    cR2y_q = ca.SX.sym('cR2y_q',N_K);  cR2y_q_lb = np.array([[q_lb]*(cR2y_q.shape[0]*cR2y_q.shape[1])]);  cR2y_q_ub = np.array([[q_ub]*(cR2y_q.shape[0]*cR2y_q.shape[1])])
    #   z-axis
    cR2z_p = ca.SX.sym('cR2z_p',N_K);  cR2z_p_lb = np.array([[p_lb]*(cR2z_p.shape[0]*cR2z_p.shape[1])]);  cR2z_p_ub = np.array([[p_ub]*(cR2z_p.shape[0]*cR2z_p.shape[1])])
    cR2z_q = ca.SX.sym('cR2z_q',N_K);  cR2z_q_lb = np.array([[q_lb]*(cR2z_q.shape[0]*cR2z_q.shape[1])]);  cR2z_q_ub = np.array([[q_ub]*(cR2z_q.shape[0]*cR2z_q.shape[1])])
    #Right Foot Contact Point 3 p and q for x, y, z
    #   x-axis
    cR3x_p = ca.SX.sym('cR3x_p',N_K);  cR3x_p_lb = np.array([[p_lb]*(cR3x_p.shape[0]*cR3x_p.shape[1])]);  cR3x_p_ub = np.array([[p_ub]*(cR3x_p.shape[0]*cR3x_p.shape[1])])
    cR3x_q = ca.SX.sym('cR3x_q',N_K);  cR3x_q_lb = np.array([[q_lb]*(cR3x_q.shape[0]*cR3x_q.shape[1])]);  cR3x_q_ub = np.array([[q_ub]*(cR3x_q.shape[0]*cR3x_q.shape[1])])
    #   y-axis
    cR3y_p = ca.SX.sym('cR3y_p',N_K);  cR3y_p_lb = np.array([[p_lb]*(cR3y_p.shape[0]*cR3y_p.shape[1])]);  cR3y_p_ub = np.array([[p_ub]*(cR3y_p.shape[0]*cR3y_p.shape[1])])
    cR3y_q = ca.SX.sym('cR3y_q',N_K);  cR3y_q_lb = np.array([[q_lb]*(cR3y_q.shape[0]*cR3y_q.shape[1])]);  cR3y_q_ub = np.array([[q_ub]*(cR3y_q.shape[0]*cR3y_q.shape[1])])
    #   z-axis
    cR3z_p = ca.SX.sym('cR3z_p',N_K);  cR3z_p_lb = np.array([[p_lb]*(cR3z_p.shape[0]*cR3z_p.shape[1])]);  cR3z_p_ub = np.array([[p_ub]*(cR3z_p.shape[0]*cR3z_p.shape[1])])
    cR3z_q = ca.SX.sym('cR3z_q',N_K);  cR3z_q_lb = np.array([[q_lb]*(cR3z_q.shape[0]*cR3z_q.shape[1])]);  cR3z_q_ub = np.array([[q_ub]*(cR3z_q.shape[0]*cR3z_q.shape[1])])    
    #Right Foot Contact Point 4 p and q for x, y, z
    #   x-axis
    cR4x_p = ca.SX.sym('cR4x_p',N_K);  cR4x_p_lb = np.array([[p_lb]*(cR4x_p.shape[0]*cR4x_p.shape[1])]);  cR4x_p_ub = np.array([[p_ub]*(cR4x_p.shape[0]*cR4x_p.shape[1])])
    cR4x_q = ca.SX.sym('cR4x_q',N_K);  cR4x_q_lb = np.array([[q_lb]*(cR4x_q.shape[0]*cR4x_q.shape[1])]);  cR4x_q_ub = np.array([[q_ub]*(cR4x_q.shape[0]*cR4x_q.shape[1])])
    #   y-axis
    cR4y_p = ca.SX.sym('cR4y_p',N_K);  cR4y_p_lb = np.array([[p_lb]*(cR4y_p.shape[0]*cR4y_p.shape[1])]);  cR4y_p_ub = np.array([[p_ub]*(cR4y_p.shape[0]*cR4y_p.shape[1])])
    cR4y_q = ca.SX.sym('cR4y_q',N_K);  cR4y_q_lb = np.array([[q_lb]*(cR4y_q.shape[0]*cR4y_q.shape[1])]);  cR4y_q_ub = np.array([[q_ub]*(cR4y_q.shape[0]*cR4y_q.shape[1])])
    #   z-axis
    cR4z_p = ca.SX.sym('cR4z_p',N_K);  cR4z_p_lb = np.array([[p_lb]*(cR4z_p.shape[0]*cR4z_p.shape[1])]);  cR4z_p_ub = np.array([[p_ub]*(cR4z_p.shape[0]*cR4z_p.shape[1])])
    cR4z_q = ca.SX.sym('cR4z_q',N_K);  cR4z_q_lb = np.array([[q_lb]*(cR4z_q.shape[0]*cR4z_q.shape[1])]);  cR4z_q_ub = np.array([[q_ub]*(cR4z_q.shape[0]*cR4z_q.shape[1])])

    #   Collect all Decision Variables
    DecisionVars = ca.vertcat(x,        y,        z,
                              xdot,     ydot,     zdot,
                              FL1x,     FL1y,     FL1z,     FL2x,     FL2y,     FL2z,     FL3x,     FL3y,     FL3z,     FL4x,     FL4y,     FL4z,
                              FR1x,     FR1y,     FR1z,     FR2x,     FR2y,     FR2z,     FR3x,     FR3y,     FR3z,     FR4x,     FR4y,     FR4z,
                              px_init,  py_init,  pz_init,
                              *px,      *py,      *pz,
                              *Ts,
                              cL1x_p,   cL1x_q,   cL1y_p,   cL1y_q,   cL1z_p,   cL1z_q,
                              cL2x_p,   cL2x_q,   cL2y_p,   cL2y_q,   cL2z_p,   cL2z_q,
                              cL3x_p,   cL3x_q,   cL3y_p,   cL3y_q,   cL3z_p,   cL3z_q,
                              cL4x_p,   cL4x_q,   cL4y_p,   cL4y_q,   cL4z_p,   cL4z_q,
                              cR1x_p,   cR1x_q,   cR1y_p,   cR1y_q,   cR1z_p,   cR1z_q,
                              cR2x_p,   cR2x_q,   cR2y_p,   cR2y_q,   cR2z_p,   cR2z_q,
                              cR3x_p,   cR3x_q,   cR3y_p,   cR3y_q,   cR3z_p,   cR3z_q,
                              cR4x_p,   cR4x_q,   cR4y_p,   cR4y_q,   cR4z_p,   cR4z_q)
    #print(DecisionVars)
    DecisionVarsShape = DecisionVars.shape

    #   Collect all lower bound and upper bound
    DecisionVars_lb = (x_lb,       y_lb,       z_lb,
                       xdot_lb,    ydot_lb,    zdot_lb,
                       FL1x_lb,    FL1y_lb,    FL1z_lb,    FL2x_lb,    FL2y_lb,    FL2z_lb,    FL3x_lb,    FL3y_lb,    FL3z_lb,    FL4x_lb,    FL4y_lb,   FL4z_lb,
                       FR1x_lb,    FR1y_lb,    FR1z_lb,    FR2x_lb,    FR2y_lb,    FR2z_lb,    FR3x_lb,    FR3y_lb,    FR3z_lb,    FR4x_lb,    FR4y_lb,   FR4z_lb,
                       px_init_lb, py_init_lb, pz_init_lb,
                       px_lb,      py_lb,      pz_lb,
                       Ts_lb,
                       cL1x_p_lb,  cL1x_q_lb,  cL1y_p_lb,  cL1y_q_lb,  cL1z_p_lb,  cL1z_q_lb,
                       cL2x_p_lb,  cL2x_q_lb,  cL2y_p_lb,  cL2y_q_lb,  cL2z_p_lb,  cL2z_q_lb,
                       cL3x_p_lb,  cL3x_q_lb,  cL3y_p_lb,  cL3y_q_lb,  cL3z_p_lb,  cL3z_q_lb,
                       cL4x_p_lb,  cL4x_q_lb,  cL4y_p_lb,  cL4y_q_lb,  cL4z_p_lb,  cL4z_q_lb,
                       cR1x_p_lb,  cR1x_q_lb,  cR1y_p_lb,  cR1y_q_lb,  cR1z_p_lb,  cR1z_q_lb,
                       cR2x_p_lb,  cR2x_q_lb,  cR2y_p_lb,  cR2y_q_lb,  cR2z_p_lb,  cR2z_q_lb,
                       cR3x_p_lb,  cR3x_q_lb,  cR3y_p_lb,  cR3y_q_lb,  cR3z_p_lb,  cR3z_q_lb,
                       cR4x_p_lb,  cR4x_q_lb,  cR4y_p_lb,  cR4y_q_lb,  cR4z_p_lb,  cR4z_q_lb)
    DecisionVars_lb = np.concatenate(DecisionVars_lb,axis=None)

    DecisionVars_ub = (x_ub,       y_ub,       z_ub,
                       xdot_ub,    ydot_ub,    zdot_ub,
                       FL1x_ub,    FL1y_ub,    FL1z_ub,    FL2x_ub,    FL2y_ub,    FL2z_ub,    FL3x_ub,    FL3y_ub,    FL3z_ub,    FL4x_ub,    FL4y_ub,    FL4z_ub,
                       FR1x_ub,    FR1y_ub,    FR1z_ub,    FR2x_ub,    FR2y_ub,    FR2z_ub,    FR3x_ub,    FR3y_ub,    FR3z_ub,    FR4x_ub,    FR4y_ub,    FR4z_ub,
                       px_init_ub, py_init_ub, pz_init_ub,
                       px_ub,      py_ub,      pz_ub,
                       Ts_ub,
                       cL1x_p_ub,  cL1x_q_ub,  cL1y_p_ub,  cL1y_q_ub,  cL1z_p_ub,  cL1z_q_ub,
                       cL2x_p_ub,  cL2x_q_ub,  cL2y_p_ub,  cL2y_q_ub,  cL2z_p_ub,  cL2z_q_ub,
                       cL3x_p_ub,  cL3x_q_ub,  cL3y_p_ub,  cL3y_q_ub,  cL3z_p_ub,  cL3z_q_ub,
                       cL4x_p_ub,  cL4x_q_ub,  cL4y_p_ub,  cL4y_q_ub,  cL4z_p_ub,  cL4z_q_ub,
                       cR1x_p_ub,  cR1x_q_ub,  cR1y_p_ub,  cR1y_q_ub,  cR1z_p_ub,  cR1z_q_ub,
                       cR2x_p_ub,  cR2x_q_ub,  cR2y_p_ub,  cR2y_q_ub,  cR2z_p_ub,  cR2z_q_ub,
                       cR3x_p_ub,  cR3x_q_ub,  cR3y_p_ub,  cR3y_q_ub,  cR3z_p_ub,  cR3z_q_ub,
                       cR4x_p_ub,  cR4x_q_ub,  cR4y_p_ub,  cR4y_q_ub,  cR4z_p_ub,  cR4z_q_ub)
    DecisionVars_ub = np.concatenate(DecisionVars_ub,axis=None)

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constrains and Running Cost
    g = [];   glb = [];   gub = []
    J = 0

    #Constraints for all knots
    #Loop over all Phases (Knots)
    for Nph in range(Nphase):
        #Decide Number of Knots
        if Nph == Nphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local  
        
        #Decide Time Step (Fixed)
        h = (PhaseDurationVec[Nph])/Nk_Local

        for Local_k_Count in range(Nk_ThisPhase):
            #Get knot number across the entire time line
            k = Nph*Nk_Local + Local_k_Count

            #------------------------------------------
            #Build useful vectors
            #   Forces
            FL1_k = ca.vertcat(FL1x[k],FL1y[k],FL1z[k]);  FL2_k = ca.vertcat(FL2x[k],FL2y[k],FL2z[k]);   FL3_k = ca.vertcat(FL3x[k],FL3y[k],FL3z[k]);   FL4_k = ca.vertcat(FL4x[k],FL4y[k],FL4z[k])
            FR1_k = ca.vertcat(FR1x[k],FR1y[k],FR1z[k]);  FR2_k = ca.vertcat(FR2x[k],FR2y[k],FR2z[k]);   FR3_k = ca.vertcat(FR3x[k],FR3y[k],FR3z[k]);   FR4_k = ca.vertcat(FR4x[k],FR4y[k],FR4z[k])
            #   CoM
            CoM_k = ca.vertcat(x[k],y[k],z[k])

            #-------------------------------------------
            #Phase dependent Constraints (CoM Kinematics and Angular Dynamics)
            #Get Step Counter
            StepCnt = Nph//3

            if GaitPattern[Nph]=='InitialDouble':
                #Special Case:
                if StepCnt == 0: #The first phase in the First STEP (Initial Double, need special care)
                    #initial support foot (the landing foot from the first phase)
                    p_init = ca.vertcat(px_init,py_init,pz_init)
                    p_init_TangentX = SurfTangentsX[0:3]
                    p_init_TangentY = SurfTangentsY[0:3]
                    p_init_Norm = SurfNorms[0:3]

                    #-----------
                    #Case 1
                    #If First Level Swing the Left, the the 0 phase (InitDouble) has p_init as the left support, PR_init as the right support
                    #(same as the double support phase of the first step)-> Left foot Moved (p_init), Right Foot stay stationary (PR_init)
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Kinematics Constraint
                    #CoM in Left (p_init)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_init, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM in Right (PR_init)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PR_init, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)

                    #CoM Height Constraint (Left p_init foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #CoM Height Constraint (Right foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PR_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact1",   P = p_init,   P_TangentX = p_init_TangentX,   P_TangentY = p_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FL1_k,             f_length = F_bound,
                                                               x_p_bar = cL1x_p[k], x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], 
                                                               z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact2",   P = p_init,   P_TangentX = p_init_TangentX,   P_TangentY = p_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FL2_k,             f_length = F_bound,
                                                               x_p_bar = cL2x_p[k], x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], 
                                                               z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact3",   P = p_init,   P_TangentX = p_init_TangentX,   P_TangentY = p_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FL3_k,             f_length = F_bound,
                                                               x_p_bar = cL3x_p[k], x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], 
                                                               z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 4                         
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact4",   P = p_init,   P_TangentX = p_init_TangentX,   P_TangentY = p_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FL4_k,             f_length = F_bound,
                                                               x_p_bar = cL4x_p[k], x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], 
                                                               z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], g = g, glb = glb, gub = gub)
                        #--------------------------
                        #Right Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact1",   P = PR_init,   P_TangentX = PR_init_TangentX,   P_TangentY = PR_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FR1_k,             f_length = F_bound,
                                                               x_p_bar = cR1x_p[k], x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], 
                                                               z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact2",   P = PR_init,   P_TangentX = PR_init_TangentX,   P_TangentY = PR_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FR2_k,             f_length = F_bound,
                                                               x_p_bar = cR2x_p[k], x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], 
                                                               z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact3",   P = PR_init,   P_TangentX = PR_init_TangentX,   P_TangentY = PR_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FR3_k,             f_length = F_bound,
                                                               x_p_bar = cR3x_p[k], x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], 
                                                               z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 4
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact4",   P = PR_init,   P_TangentX = PR_init_TangentX,   P_TangentY = PR_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FR4_k,             f_length = F_bound,
                                                               x_p_bar = cR4x_p[k], x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], 
                                                               z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], g = g, glb = glb, gub = gub)
                    #Unilateral Constraint
                    #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the unilateral constraint on p_init
                    #Left Foot force (p_init)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the PR_init
                    #Right Foot (PR_init)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the friction cone constraint on p_init
                    #Left Foot (p_init)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the PR_init
                    #Right Foot (PR_init)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    
                    #--------------------
                    #Case 2
                    #If First Level Swing the Right, the the 0 phase (InitDouble) has p_init as the Right support, PL_init as the Left support
                    #(same as the double support phase of the first step)-> Right foot Moved (p_init), Left Foot stay stationary (PL_init)
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Kinematics Constraint
                    #CoM in Left (PL_init)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PL_init, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM in Right (p_init)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_init, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                
                    #CoM Height Constraint (Left PL_init foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PL_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #CoM Height Constraint (Right p_init foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics
                    if k<N_K-1:
                        #------------------------------
                        #Left Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact1",   P = PL_init,   P_TangentX = PL_init_TangentX,   P_TangentY = PL_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FL1_k,             f_length = F_bound,
                                                               x_p_bar = cL1x_p[k], x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], 
                                                               z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact2",   P = PL_init,   P_TangentX = PL_init_TangentX,   P_TangentY = PL_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FL2_k,             f_length = F_bound,
                                                               x_p_bar = cL2x_p[k], x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], 
                                                               z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact3",   P = PL_init,   P_TangentX = PL_init_TangentX,   P_TangentY = PL_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FL3_k,             f_length = F_bound,
                                                               x_p_bar = cL3x_p[k], x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], 
                                                               z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 4                         
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact4",   P = PL_init,   P_TangentX = PL_init_TangentX,   P_TangentY = PL_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FL4_k,             f_length = F_bound,
                                                               x_p_bar = cL4x_p[k], x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], 
                                                               z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], g = g, glb = glb, gub = gub)
                        #--------------------------
                        #Right Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact1",   P = p_init,   P_TangentX = p_init_TangentX,   P_TangentY = p_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FR1_k,             f_length = F_bound,
                                                               x_p_bar = cR1x_p[k], x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], 
                                                               z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact2",   P = p_init,   P_TangentX = p_init_TangentX,   P_TangentY = p_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FR2_k,             f_length = F_bound,
                                                               x_p_bar = cR2x_p[k], x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], 
                                                               z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact3",   P = p_init,   P_TangentX = p_init_TangentX,   P_TangentY = p_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FR3_k,             f_length = F_bound,
                                                               x_p_bar = cR3x_p[k], x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], 
                                                               z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 4
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Contact4",   P = p_init,   P_TangentX = p_init_TangentX,   P_TangentY = p_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FR4_k,             f_length = F_bound,
                                                               x_p_bar = cR4x_p[k], x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], 
                                                               z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], g = g, glb = glb, gub = gub)
                    #Unilateral Constraint
                    #Left Foot (PL_init)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                    #Right Foot (p_init)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    
                    #Friction Cone Constraint
                    #Left Foot (PL_init)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #Right Foot (p_init)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)

                elif StepCnt > 0:#Other Cases
                    #Get contact location and Terrain Tangents and Norms
                    if StepCnt == 1: #Step 1 needs special care (NOTE: Step Count Start from 0)
                        #Previous Step
                        p_previous = ca.vertcat(px_init,py_init,pz_init)
                        p_previous_TangentX = SurfTangentsX[0:3];   p_previous_TangentY = SurfTangentsY[0:3]
                        p_previous_Norm = SurfNorms[0:3]

                        #Current Step
                        #In second level, Surfaces index is Step Vector Index (for px, py, pz, here is StepCnt-1) + 1
                        #For Intial Double Support, previous step is StepNum - 2, current step is StepNum - 1
                        #but index for the tangents and norm need to +1 as StepNum (in the second level) + 1 = StepNum in the entire horizon
                        p_current = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_current_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3];   p_current_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3] 
                        p_current_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    else: #Like Step 2, 3, 4 .....
                        #For Intial Double Support, previous step is StepNum - 2, current step is StepNum - 1
                        #but index for the tangents and norm need to +1 as StepNum (in the second level) + 1 = StepNum in the entire horizon
                        #For Initial Double Support, the contact config is the same as the double support phase of the previous step, where p_current is the landed foot
                        #p_previous is the non-moving foot
                        p_previous = ca.vertcat(px[StepCnt-2],py[StepCnt-2],pz[StepCnt-2])
                        p_previous_TangentX = SurfTangentsX[(StepCnt-1)*3:(StepCnt-1)*3+3];   p_previous_TangentY = SurfTangentsY[(StepCnt-1)*3:(StepCnt-1)*3+3]
                        p_previous_Norm = SurfNorms[(StepCnt-1)*3:(StepCnt-1)*3+3]

                        p_current = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_current_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3];   p_current_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                        p_current_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    #Give Constraint according to even and odd steps
                    if StepCnt%2 == 0: #Even Numbers of Footsteps
                        #----------
                        #Case 1
                        #If the first level swing the Left, then the Even Number of Steps in the Intial Double support phase have p_current as Left Support (Landed), p_previous as Right Support (Stationary)
                        #(same as the double support phase of the first step)->Left foot Moved (p_current), Right Foot stay stationary p_previous)
                        SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                        #CoM in the Left (p_current)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                        #CoM in the Right (p_previous)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Left p_current foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Right p_previous foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #Angular Dynamics (Ponton)
                        if k<N_K-1: #double check the knot number is valid
                            #------------------------------
                            #Left Foot Contact Point 1
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact1",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL1_k,             f_length = F_bound,
                                                                x_p_bar = cL1x_p[k], x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], 
                                                                z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], g = g, glb = glb, gub = gub)
                            #Left Foot Contact Point 2
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact2",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL2_k,             f_length = F_bound,
                                                                x_p_bar = cL2x_p[k], x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], 
                                                                z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], g = g, glb = glb, gub = gub)
                            #Left Foot Contact Point 3
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact3",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL3_k,             f_length = F_bound,
                                                                x_p_bar = cL3x_p[k], x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], 
                                                                z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], g = g, glb = glb, gub = gub)
                            #Left Foot Contact Point 4                         
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact4",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL4_k,             f_length = F_bound,
                                                                x_p_bar = cL4x_p[k], x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], 
                                                                z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], g = g, glb = glb, gub = gub)
                            #--------------------------
                            #Right Foot Contact Point 1
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact1",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR1_k,             f_length = F_bound,
                                                                x_p_bar = cR1x_p[k], x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], 
                                                                z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], g = g, glb = glb, gub = gub)
                            #Right Foot Contact Point 2
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact2",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR2_k,             f_length = F_bound,
                                                                x_p_bar = cR2x_p[k], x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], 
                                                                z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], g = g, glb = glb, gub = gub)
                            #Right Foot Contact Point 3
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact3",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR3_k,             f_length = F_bound,
                                                                x_p_bar = cR3x_p[k], x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], 
                                                                z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], g = g, glb = glb, gub = gub)
                            #Right Foot Contact Point 4
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact4",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR4_k,             f_length = F_bound,
                                                                x_p_bar = cR4x_p[k], x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], 
                                                                z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], g = g, glb = glb, gub = gub)
                        #Unilateral Constraint
                        #Left Foot (p_current)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        #Right Foot (p_previous)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        #Friction Cone Constraint
                        #Left Foot (p_current)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        #Right Foot (p_previous)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        
                        #---------
                        #Case 2
                        #If the first level swing the Right, then the Even Number of Steps in the Intial Double support phase have p_current as Right Support (Landed), 
                        #(same as the double support phase of the first step) -> Right Moved (p_current), Left stationary (p_previous)
                        SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                        #CoM in the Left (p_previous)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                        #CoM in the Right (p_current)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Left p_previous foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Right p_current foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #Angular Dynamics (Ponton)
                        if k<N_K-1: #double check the knot number is valid
                            #------------------------------
                            #Left Foot Contact Point 1
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact1",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL1_k,             f_length = F_bound,
                                                                x_p_bar = cL1x_p[k], x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], 
                                                                z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], g = g, glb = glb, gub = gub)
                            #Left Foot Contact Point 2
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact2",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL2_k,             f_length = F_bound,
                                                                x_p_bar = cL2x_p[k], x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], 
                                                                z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], g = g, glb = glb, gub = gub)
                            #Left Foot Contact Point 3
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact3",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL3_k,             f_length = F_bound,
                                                                x_p_bar = cL3x_p[k], x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], 
                                                                z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], g = g, glb = glb, gub = gub)
                            #Left Foot Contact Point 4                         
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact4",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL4_k,             f_length = F_bound,
                                                                x_p_bar = cL4x_p[k], x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], 
                                                                z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], g = g, glb = glb, gub = gub)
                            #--------------------------
                            #Right Foot Contact Point 1
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact1",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR1_k,             f_length = F_bound,
                                                                x_p_bar = cR1x_p[k], x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], 
                                                                z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], g = g, glb = glb, gub = gub)
                            #Right Foot Contact Point 2
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact2",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR2_k,             f_length = F_bound,
                                                                x_p_bar = cR2x_p[k], x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], 
                                                                z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], g = g, glb = glb, gub = gub)
                            #Right Foot Contact Point 3
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact3",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR3_k,             f_length = F_bound,
                                                                x_p_bar = cR3x_p[k], x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], 
                                                                z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], g = g, glb = glb, gub = gub)
                            #Right Foot Contact Point 4
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact4",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR4_k,             f_length = F_bound,
                                                                x_p_bar = cR4x_p[k], x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], 
                                                                z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], g = g, glb = glb, gub = gub)
                        #Unilateral Constraint
                        #Left foot (p_previous)                         
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        #Right foot (p_current)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        #Friction Cone Constraint
                        #Left foot (p_previous)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        #Right Foot (p_current)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    elif StepCnt%2 == 1: #Odd Number of Steps
                        #---------
                        #Case 1
                        #If the first level swing the Left, then the Odd Number of Steps in the Intial Double support phase (Swing Right) have p_current as Right Support (Landed), p_previous as Left Support (Stationary)
                        #(same as the double support phase of the first step) -> Right Moved (p_current), Left Stay Fixed (p_previous)
                        SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                        #CoM in the Left (p_previous)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                        #CoM in the Right (p_current)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Left p_previous foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Right p_current foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #Angular Dynamics (Ponton)
                        if k<N_K-1: #double check the knot number is valid
                            #------------------------------
                            #Left Foot Contact Point 1
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact1",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL1_k,             f_length = F_bound,
                                                                x_p_bar = cL1x_p[k], x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], 
                                                                z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], g = g, glb = glb, gub = gub)
                            #Left Foot Contact Point 2
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact2",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL2_k,             f_length = F_bound,
                                                                x_p_bar = cL2x_p[k], x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], 
                                                                z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], g = g, glb = glb, gub = gub)
                            #Left Foot Contact Point 3
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact3",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL3_k,             f_length = F_bound,
                                                                x_p_bar = cL3x_p[k], x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], 
                                                                z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], g = g, glb = glb, gub = gub)
                            #Left Foot Contact Point 4                         
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact4",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL4_k,             f_length = F_bound,
                                                                x_p_bar = cL4x_p[k], x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], 
                                                                z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], g = g, glb = glb, gub = gub)
                            #--------------------------
                            #Right Foot Contact Point 1
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact1",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR1_k,             f_length = F_bound,
                                                                x_p_bar = cR1x_p[k], x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], 
                                                                z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], g = g, glb = glb, gub = gub)
                            #Right Foot Contact Point 2
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact2",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR2_k,             f_length = F_bound,
                                                                x_p_bar = cR2x_p[k], x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], 
                                                                z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], g = g, glb = glb, gub = gub)
                            #Right Foot Contact Point 3
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact3",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR3_k,             f_length = F_bound,
                                                                x_p_bar = cR3x_p[k], x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], 
                                                                z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], g = g, glb = glb, gub = gub)
                            #Right Foot Contact Point 4
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact4",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR4_k,             f_length = F_bound,
                                                                x_p_bar = cR4x_p[k], x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], 
                                                                z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], g = g, glb = glb, gub = gub)
                        #Unilateral Constraint
                        #Left foot (p_previous)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        #Right foot (p_current)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        #Friction Cone Constraint
                        #Left foot (p_previous)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        #Right Foot (p_current)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        
                        #-----
                        #Case 2
                        #If the first level swing the Right, then the Odd Number of Steps in the Intial Double support phase have p_current as Left Support (Landed), 
                        #(same as the double support phase of the first step) -> Left Moved (p_current), Right stationary (p_previous)
                        SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                        #CoM in the Left (p_current)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                        #CoM in the Right (p_previous)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Left p_current foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Right p_previous foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #Angular Dynamics (Ponton)
                        if k<N_K-1: #double check the knot number is valid
                            #------------------------------
                            #Left Foot Contact Point 1
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact1",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL1_k,             f_length = F_bound,
                                                                x_p_bar = cL1x_p[k], x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], 
                                                                z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], g = g, glb = glb, gub = gub)
                            #Left Foot Contact Point 2
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact2",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL2_k,             f_length = F_bound,
                                                                x_p_bar = cL2x_p[k], x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], 
                                                                z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], g = g, glb = glb, gub = gub)
                            #Left Foot Contact Point 3
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact3",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL3_k,             f_length = F_bound,
                                                                x_p_bar = cL3x_p[k], x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], 
                                                                z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], g = g, glb = glb, gub = gub)
                            #Left Foot Contact Point 4                         
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact4",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL4_k,             f_length = F_bound,
                                                                x_p_bar = cL4x_p[k], x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], 
                                                                z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], g = g, glb = glb, gub = gub)
                            #--------------------------
                            #Right Foot Contact Point 1
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact1",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR1_k,             f_length = F_bound,
                                                                x_p_bar = cR1x_p[k], x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], 
                                                                z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], g = g, glb = glb, gub = gub)
                            #Right Foot Contact Point 2
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact2",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR2_k,             f_length = F_bound,
                                                                x_p_bar = cR2x_p[k], x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], 
                                                                z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], g = g, glb = glb, gub = gub)
                            #Right Foot Contact Point 3
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact3",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR3_k,             f_length = F_bound,
                                                                x_p_bar = cR3x_p[k], x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], 
                                                                z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], g = g, glb = glb, gub = gub)
                            #Right Foot Contact Point 4
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Contact4",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR4_k,             f_length = F_bound,
                                                                x_p_bar = cR4x_p[k], x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], 
                                                                z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], g = g, glb = glb, gub = gub)
                        #Unilateral Constraint
                        #Left foot (p_current)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        #Right foot (p_previous)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        #Friction Cone Constraint
                        #Left foot (p_current)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        #right foot (p_previous)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)

            elif GaitPattern[Nph]== 'Swing':
                #Get contact location
                #In the swing phase, the stance leg is the landing foot of the previous step (Step Number - 1), 
                #but index for the tangents and norm need to +1 as StepNum (in the second level) + 1 = StepNum in the entire horizon
                if StepCnt == 0:#Special Case for the First Step (NOTE:Step 0)
                    p_stance = ca.vertcat(px_init,py_init,pz_init)
                    p_stance_TangentX = SurfTangentsX[0:3]
                    p_stance_TangentY = SurfTangentsY[0:3]
                    p_stance_Norm = SurfNorms[0:3]

                else: #For other Steps, indexed as 1,2,3,4
                    p_stance = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                    p_stance_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                    p_stance_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                    p_stance_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                #Give Constraint according to even and odd steps
                if StepCnt%2 == 0: #Even Number of Steps
                    #------
                    #Case 1
                    #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Left foot is the stance foot

                    #Kinematics Constraint
                    #CoM in the Left (p_stance)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM Height Constraint (Left p_stance foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact1",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL1_k,             f_length = F_bound,
                                                            x_p_bar = cL1x_p[k], x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], 
                                                            z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact2",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL2_k,             f_length = F_bound,
                                                            x_p_bar = cL2x_p[k], x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], 
                                                            z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact3",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL3_k,             f_length = F_bound,
                                                            x_p_bar = cL3x_p[k], x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], 
                                                            z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 4                         
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact4",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL4_k,             f_length = F_bound,
                                                            x_p_bar = cL4x_p[k], x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], 
                                                            z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], g = g, glb = glb, gub = gub)
                    #Zero Ponton Terms
                    #For Right Foot Contact Points
                    ponton_term_vec = ca.vertcat(cR1x_p[k], cR1x_q[k], cR1y_p[k], cR1y_q[k], cR1z_p[k], cR1z_q[k],
                                                 cR2x_p[k], cR2x_q[k], cR2y_p[k], cR2y_q[k], cR2z_p[k], cR2z_q[k],
                                                 cR3x_p[k], cR3x_q[k], cR3y_p[k], cR3y_q[k], cR3z_p[k], cR3z_q[k],
                                                 cR4x_p[k], cR4x_q[k], cR4y_p[k], cR4y_q[k], cR4z_p[k], cR4z_q[k])
                    g.append(ca.if_else(SwingLegFlag,ponton_term_vec,np.zeros(4*6)))
                    glb.append(np.zeros(4*6))
                    gub.append(np.zeros(4*6))

                    #Zero Forces (Right Foot)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraints on Left Foot p_stance
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint on Left Foot p_stance
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)

                    #--------------
                    #Case 2
                    #If First Level Swing the Right, then the second level Even Number Phases (the first Phase) Swing the Left
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Right foot is the stance foot

                    #Kinematics Constraint
                    #CoM in the Right (p_stance)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                    #CoM Height Constraint (Right p_stance foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact1",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR1_k,             f_length = F_bound,
                                                            x_p_bar = cR1x_p[k], x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], 
                                                            z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact2",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR2_k,             f_length = F_bound,
                                                            x_p_bar = cR2x_p[k], x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], 
                                                            z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact3",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR3_k,             f_length = F_bound,
                                                            x_p_bar = cR3x_p[k], x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], 
                                                            z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 4                         
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact4",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR4_k,             f_length = F_bound,
                                                            x_p_bar = cR4x_p[k], x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], 
                                                            z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], g = g, glb = glb, gub = gub)
                    #Zero Ponton Terms
                    #For Left Foot Contact Points
                    ponton_term_vec = ca.vertcat(cL1x_p[k], cL1x_q[k], cL1y_p[k], cL1y_q[k], cL1z_p[k], cL1z_q[k],
                                                 cL2x_p[k], cL2x_q[k], cL2y_p[k], cL2y_q[k], cL2z_p[k], cL2z_q[k],
                                                 cL3x_p[k], cL3x_q[k], cL3y_p[k], cL3y_q[k], cL3z_p[k], cL3z_q[k],
                                                 cL4x_p[k], cL4x_q[k], cL4y_p[k], cL4y_q[k], cL4z_p[k], cL4z_q[k])
                    g.append(ca.if_else(SwingLegFlag,ponton_term_vec,np.zeros(4*6)))
                    glb.append(np.zeros(4*6))
                    gub.append(np.zeros(4*6))

                    #Zero Forces (Left Foot)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraints on Right Foot p_stance
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint on Right Foot p_stance
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)

                elif StepCnt%2 == 1: #Odd Number of Steps
                    #--------
                    #Case 1
                    #If First Level Swing the Left, then the second level Odd Number Steps Swing the Left
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Right Foot is the stance foot

                    #Kinematics Constraint
                    #CoM in the Right (p_stance)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_stance foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact1",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR1_k,             f_length = F_bound,
                                                            x_p_bar = cR1x_p[k], x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], 
                                                            z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact2",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR2_k,             f_length = F_bound,
                                                            x_p_bar = cR2x_p[k], x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], 
                                                            z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact3",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR3_k,             f_length = F_bound,
                                                            x_p_bar = cR3x_p[k], x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], 
                                                            z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 4                         
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact4",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR4_k,             f_length = F_bound,
                                                            x_p_bar = cR4x_p[k], x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], 
                                                            z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], g = g, glb = glb, gub = gub)
                    #Zero Ponton Terms
                    #For Left Foot Contact Points
                    ponton_term_vec = ca.vertcat(cL1x_p[k], cL1x_q[k], cL1y_p[k], cL1y_q[k], cL1z_p[k], cL1z_q[k],
                                                 cL2x_p[k], cL2x_q[k], cL2y_p[k], cL2y_q[k], cL2z_p[k], cL2z_q[k],
                                                 cL3x_p[k], cL3x_q[k], cL3y_p[k], cL3y_q[k], cL3z_p[k], cL3z_q[k],
                                                 cL4x_p[k], cL4x_q[k], cL4y_p[k], cL4y_q[k], cL4z_p[k], cL4z_q[k])
                    g.append(ca.if_else(SwingLegFlag,ponton_term_vec,np.zeros(4*6)))
                    glb.append(np.zeros(4*6))
                    gub.append(np.zeros(4*6))

                    #Zero Forces (Left Foot)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraints on Right Foot p_stance
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint on Right Foot p_stance
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)

                    #-------
                    #Case 2
                    #If First Level Swing the Right, then the second level Odd Number Steps Swing the Right
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Left foot is the stance foot

                    #Kinematics Constraint
                    #CoM in the Left (p_stance)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Left p_stance foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact1",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL1_k,             f_length = F_bound,
                                                            x_p_bar = cL1x_p[k], x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], 
                                                            z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact2",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL2_k,             f_length = F_bound,
                                                            x_p_bar = cL2x_p[k], x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], 
                                                            z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact3",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL3_k,             f_length = F_bound,
                                                            x_p_bar = cL3x_p[k], x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], 
                                                            z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 4                         
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact4",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL4_k,             f_length = F_bound,
                                                            x_p_bar = cL4x_p[k], x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], 
                                                            z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], g = g, glb = glb, gub = gub)
                    #Zero Ponton Terms
                    #For Right Foot Contact Points
                    ponton_term_vec = ca.vertcat(cR1x_p[k], cR1x_q[k], cR1y_p[k], cR1y_q[k], cR1z_p[k], cR1z_q[k],
                                                 cR2x_p[k], cR2x_q[k], cR2y_p[k], cR2y_q[k], cR2z_p[k], cR2z_q[k],
                                                 cR3x_p[k], cR3x_q[k], cR3y_p[k], cR3y_q[k], cR3z_p[k], cR3z_q[k],
                                                 cR4x_p[k], cR4x_q[k], cR4y_p[k], cR4y_q[k], cR4z_p[k], cR4z_q[k])
                    g.append(ca.if_else(SwingLegFlag,ponton_term_vec,np.zeros(4*6)))
                    glb.append(np.zeros(4*6))
                    gub.append(np.zeros(4*6))

                    #Zero Forces (Right Foot)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, g = g, glb = glb, gub = gub)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraints on Left Foot p_stance
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint on Left Foot p_stance
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)

            elif GaitPattern[Nph]=='DoubleSupport':
                #Get contact location
                #In the Double Support Phase, the p_stationary is the foot is the un-moved foot during StepNum (in second level)
                #the p_land is the landing/moving foot during StepNum (in the second level)
                if StepCnt == 0: #Special Case for the First Step (NOTE: Step 0)
                    p_stationary = ca.vertcat(px_init,py_init,pz_init)
                    p_stationary_TangentX = SurfTangentsX[0:3]
                    p_stationary_TangentY = SurfTangentsY[0:3]
                    p_stationary_Norm = SurfNorms[0:3]

                    p_land = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                    p_land_TangentX = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_TangentY = SurfTangentsY[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_Norm = SurfNorms[(StepCnt+1)*3:(StepCnt+1)*3+3]
            
                else: #For other steps, indexed as 1,2,3,4
                    #In the Double Support Phase, the p_stationary is the foot is the un-moved foot during StepNum (in second level), \
                    #which is StepCnt - 1 (the moving foot in the previous step)
                    #the p_land is the landing/moving foot during StepNum (in the second level) --- StepCnt
                    #The StepCnt + 1 is the StepNum in the entire horizon (for getting terrain tangents and norm)
                    p_stationary = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                    p_stationary_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                    p_stationary_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                    p_stationary_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    p_land = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                    p_land_TangentX = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_TangentY = SurfTangentsY[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_Norm = SurfNorms[(StepCnt+1)*3:(StepCnt+1)*3+3]

                #Give Constraint according to even and odd steps
                if StepCnt%2 == 0: #Even Number of Steps
                    #-----------
                    #Case 1
                    #If First Level Swing the Left, then the second level Even Steps Swing the Right
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Left Foot is stationary
                    #Right Foot is the Land

                    #Kinemactics Constraint
                    #CoM in the Left (p_stationary)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM in the Right (p_land)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                    
                    #   CoM Height Constraint (Left p_stationary foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_land foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact1",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL1_k,             f_length = F_bound,
                                                            x_p_bar = cL1x_p[k], x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], 
                                                            z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact2",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL2_k,             f_length = F_bound,
                                                            x_p_bar = cL2x_p[k], x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], 
                                                            z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact3",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL3_k,             f_length = F_bound,
                                                            x_p_bar = cL3x_p[k], x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], 
                                                            z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 4                         
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact4",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL4_k,             f_length = F_bound,
                                                            x_p_bar = cL4x_p[k], x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], 
                                                            z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], g = g, glb = glb, gub = gub)
                        #--------------------------
                        #Right Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact1",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR1_k,             f_length = F_bound,
                                                            x_p_bar = cR1x_p[k], x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], 
                                                            z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact2",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR2_k,             f_length = F_bound,
                                                            x_p_bar = cR2x_p[k], x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], 
                                                            z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact3",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR3_k,             f_length = F_bound,
                                                            x_p_bar = cR3x_p[k], x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], 
                                                            z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 4
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact4",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR4_k,             f_length = F_bound,
                                                            x_p_bar = cR4x_p[k], x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], 
                                                            z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], g = g, glb = glb, gub = gub)
                    #Unilateral Constraint
                    #Left foot obey the unilateral constraint on p_stationary
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the Stationary foot p_land
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #Left foot obey the friction cone constraint on p_stationary
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the on p_land
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)

                    #------------
                    #Case 2
                    #If First Level Swing the Right, then the second level Even Steps Swing the Left
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Right Foot is stationary
                    #Left Foot is the Land

                    #Kinemactics Constraint
                    #CoM in the Left (p_land)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM in the Right (p_stationary)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                    
                    #   CoM Height Constraint (Left p_land foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_stationary foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact1",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL1_k,             f_length = F_bound,
                                                            x_p_bar = cL1x_p[k], x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], 
                                                            z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact2",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL2_k,             f_length = F_bound,
                                                            x_p_bar = cL2x_p[k], x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], 
                                                            z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact3",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL3_k,             f_length = F_bound,
                                                            x_p_bar = cL3x_p[k], x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], 
                                                            z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 4                         
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact4",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL4_k,             f_length = F_bound,
                                                            x_p_bar = cL4x_p[k], x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], 
                                                            z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], g = g, glb = glb, gub = gub)
                        #--------------------------
                        #Right Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact1",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR1_k,             f_length = F_bound,
                                                            x_p_bar = cR1x_p[k], x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], 
                                                            z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact2",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR2_k,             f_length = F_bound,
                                                            x_p_bar = cR2x_p[k], x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], 
                                                            z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact3",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR3_k,             f_length = F_bound,
                                                            x_p_bar = cR3x_p[k], x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], 
                                                            z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 4
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact4",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR4_k,             f_length = F_bound,
                                                            x_p_bar = cR4x_p[k], x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], 
                                                            z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], g = g, glb = glb, gub = gub)
                    #Unilateral Constraint
                    #Left foot obey the unilateral constraint on p_land
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the Stationary foot p_stationary
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #Left foot obey the friction cone constraint on p_land
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the on p_stationary
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)

                elif StepCnt%2 == 1:#Odd Number of Steps
                    #------
                    #Case 1
                    #If First Level Swing the Left, then the second level Odd Steps Swing the Left
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Right Foot is the Stationary
                    #Left Foot is the Land

                    #Kinemactics Constraint
                    #CoM in the Left (p_land)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM in the Right (p_stationary)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Left p_land foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_stationary foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact1",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL1_k,             f_length = F_bound,
                                                            x_p_bar = cL1x_p[k], x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], 
                                                            z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact2",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL2_k,             f_length = F_bound,
                                                            x_p_bar = cL2x_p[k], x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], 
                                                            z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact3",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL3_k,             f_length = F_bound,
                                                            x_p_bar = cL3x_p[k], x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], 
                                                            z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 4                         
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact4",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL4_k,             f_length = F_bound,
                                                            x_p_bar = cL4x_p[k], x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], 
                                                            z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], g = g, glb = glb, gub = gub)
                        #--------------------------
                        #Right Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact1",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR1_k,             f_length = F_bound,
                                                            x_p_bar = cR1x_p[k], x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], 
                                                            z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact2",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR2_k,             f_length = F_bound,
                                                            x_p_bar = cR2x_p[k], x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], 
                                                            z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact3",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR3_k,             f_length = F_bound,
                                                            x_p_bar = cR3x_p[k], x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], 
                                                            z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 4
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact4",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR4_k,             f_length = F_bound,
                                                            x_p_bar = cR4x_p[k], x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], 
                                                            z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], g = g, glb = glb, gub = gub)
                    #Unilateral Constraint
                    #Left foot obey the unilateral constraint on p_land
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the Stationary foot p_stationary
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #Left foot obey the friction cone constraint on p_land
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the on p_stationary
                    g, glb, gub = FrictionCone(SwingLegIndicator = ParaLeftSwingFlag, F_k = FR1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = ParaLeftSwingFlag, F_k = FR2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = ParaLeftSwingFlag, F_k = FR3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = ParaLeftSwingFlag, F_k = FR4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
  
                    #------
                    #Case 2
                    #If First Level Swing the Right, then the second level Odd Steps Swing the Right
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Left Foot is the stationary
                    #Right Foot is the Land

                    #Kinematics Constraint
                    #CoM in the Left (p_stationary)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM in the Right (p_land)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Left p_stationary foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_land foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact1",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL1_k,             f_length = F_bound,
                                                            x_p_bar = cL1x_p[k], x_q_bar = cL1x_q[k], y_p_bar = cL1y_p[k], y_q_bar = cL1y_q[k], 
                                                            z_p_bar = cL1z_p[k], z_q_bar = cL1z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact2",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL2_k,             f_length = F_bound,
                                                            x_p_bar = cL2x_p[k], x_q_bar = cL2x_q[k], y_p_bar = cL2y_p[k], y_q_bar = cL2y_q[k], 
                                                            z_p_bar = cL2z_p[k], z_q_bar = cL2z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact3",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL3_k,             f_length = F_bound,
                                                            x_p_bar = cL3x_p[k], x_q_bar = cL3x_q[k], y_p_bar = cL3y_p[k], y_q_bar = cL3y_q[k], 
                                                            z_p_bar = cL3z_p[k], z_q_bar = cL3z_q[k], g = g, glb = glb, gub = gub)
                        #Left Foot Contact Point 4                         
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact4",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL4_k,             f_length = F_bound,
                                                            x_p_bar = cL4x_p[k], x_q_bar = cL4x_q[k], y_p_bar = cL4y_p[k], y_q_bar = cL4y_q[k], 
                                                            z_p_bar = cL4z_p[k], z_q_bar = cL4z_q[k], g = g, glb = glb, gub = gub)
                        #--------------------------
                        #Right Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact1",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR1_k,             f_length = F_bound,
                                                            x_p_bar = cR1x_p[k], x_q_bar = cR1x_q[k], y_p_bar = cR1y_p[k], y_q_bar = cR1y_q[k], 
                                                            z_p_bar = cR1z_p[k], z_q_bar = cR1z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 2
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact2",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR2_k,             f_length = F_bound,
                                                            x_p_bar = cR2x_p[k], x_q_bar = cR2x_q[k], y_p_bar = cR2y_p[k], y_q_bar = cR2y_q[k], 
                                                            z_p_bar = cR2z_p[k], z_q_bar = cR2z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 3
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact3",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR3_k,             f_length = F_bound,
                                                            x_p_bar = cR3x_p[k], x_q_bar = cR3x_q[k], y_p_bar = cR3y_p[k], y_q_bar = cR3y_q[k], 
                                                            z_p_bar = cR3z_p[k], z_q_bar = cR3z_q[k], g = g, glb = glb, gub = gub)
                        #Right Foot Contact Point 4
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Contact4",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR4_k,             f_length = F_bound,
                                                            x_p_bar = cR4x_p[k], x_q_bar = cR4x_q[k], y_p_bar = cR4y_p[k], y_q_bar = cR4y_q[k], 
                                                            z_p_bar = cR4z_p[k], z_q_bar = cR4z_q[k], g = g, glb = glb, gub = gub)
                    #Unilateral Constraint
                    #Left foot obey the unilateral constraint on p_stationary
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the Stationary foot p_land
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #Left foot obey the friction cone constraint on p_stationary
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL1_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL2_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL3_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL4_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the on p_land
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR1_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR2_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR3_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR4_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)

                    #-------------------------------------
            else:
                raise Exception("Unknown Phase Name")                   

            # #-------------------------------------
            # #Dynamics Constraint
            if k < N_K - 1: #N_K - 1 the enumeration of the last knot, -1 the knot before the last knot
                #First-order Dynamics CoM x, y, z
                g, glb, gub = First_Order_Integrator(next_state = x[k+1], cur_state = x[k], cur_derivative = xdot[k], h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = y[k+1], cur_state = y[k], cur_derivative = ydot[k], h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = z[k+1], cur_state = z[k], cur_derivative = zdot[k], h = h, g = g, glb = glb, gub = gub)
                #First-order Dynamics CoMdot x, y, z
                g, glb, gub = First_Order_Integrator(next_state = xdot[k+1], cur_state = xdot[k], 
                                                     cur_derivative = FL1x[k]/m + FL2x[k]/m + FL3x[k]/m + FL4x[k]/m + FR1x[k]/m + FR2x[k]/m + FR3x[k]/m + FR4x[k]/m, 
                                                     h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = ydot[k+1], cur_state = ydot[k], 
                                                     cur_derivative = FL1y[k]/m + FL2y[k]/m + FL3y[k]/m + FL4y[k]/m + FR1y[k]/m + FR2y[k]/m + FR3y[k]/m + FR4y[k]/m, 
                                                     h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = zdot[k+1], cur_state = zdot[k], 
                                                     cur_derivative = FL1z[k]/m + FL2z[k]/m + FL3z[k]/m + FL4z[k]/m + FR1z[k]/m + FR2z[k]/m + FR3z[k]/m + FR4z[k]/m - G, 
                                                     h = h, g = g, glb = glb, gub = gub)

            #Cost Terms
            if k < N_K - 1:
                #with Acceleration
                J = J + h*(FL1x[k]/m + FL2x[k]/m + FL3x[k]/m + FL4x[k]/m + FR1x[k]/m + FR2x[k]/m + FR3x[k]/m + FR4x[k]/m)**2 + \
                        h*(FL1y[k]/m + FL2y[k]/m + FL3y[k]/m + FL4y[k]/m + FR1y[k]/m + FR2y[k]/m + FR3y[k]/m + FR4y[k]/m)**2 + \
                        h*(FL1z[k]/m + FL2z[k]/m + FL3z[k]/m + FL4z[k]/m + FR1z[k]/m + FR2z[k]/m + FR3z[k]/m + FR4z[k]/m - G)**2
                #with Angular Momentum Rate (Ponton)
                J = J + h*(cL1x_p[k]**2 + cL1x_q[k]**2 + cL1y_p[k]**2 + cL1y_q[k]**2 + cL1z_p[k]**2 + cL1z_q[k]**2 + \
                           cL2x_p[k]**2 + cL2x_q[k]**2 + cL2y_p[k]**2 + cL2y_q[k]**2 + cL2z_p[k]**2 + cL2z_q[k]**2 + \
                           cL3x_p[k]**2 + cL3x_q[k]**2 + cL3y_p[k]**2 + cL3y_q[k]**2 + cL3z_p[k]**2 + cL3z_q[k]**2 + \
                           cL4x_p[k]**2 + cL4x_q[k]**2 + cL4y_p[k]**2 + cL4y_q[k]**2 + cL4z_p[k]**2 + cL4z_q[k]**2 + \
                           cR1x_p[k]**2 + cR1x_q[k]**2 + cR1y_p[k]**2 + cR1y_q[k]**2 + cR1z_p[k]**2 + cR1z_q[k]**2 + \
                           cR2x_p[k]**2 + cR2x_q[k]**2 + cR2y_p[k]**2 + cR2y_q[k]**2 + cR2z_p[k]**2 + cR2z_q[k]**2 + \
                           cR3x_p[k]**2 + cR3x_q[k]**2 + cR3y_p[k]**2 + cR3y_q[k]**2 + cR3z_p[k]**2 + cR3z_q[k]**2 + \
                           cR4x_p[k]**2 + cR4x_q[k]**2 + cR4y_p[k]**2 + cR4y_q[k]**2 + cR4z_p[k]**2 + cR4z_q[k]**2)
    
    #-------------------------------------
    #Relative Footstep Constraint
    for step_cnt in range(Nsteps):
        if step_cnt == 0:
            #!!!!!!Pass from the first Level!!!!!!
            P_k_current = ca.vertcat(px_init,py_init,pz_init) #ca.vertcat(px[-1],py[-1],pz[-1])
            #!!!!!!
            P_k_next = ca.vertcat(px[step_cnt],py[step_cnt],pz[step_cnt])
        else:
            P_k_current = ca.vertcat(px[step_cnt-1],py[step_cnt-1],pz[step_cnt-1])
            P_k_next = ca.vertcat(px[step_cnt],py[step_cnt],pz[step_cnt])

        if step_cnt%2 == 0: #even number steps
            #----
            #Case 1
            #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right
            SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
            #Right Foot is landing (p_next), Left foot is stationary (p_current)
            #Relative Swing Foot Location - Landing (p_next/Right) in Stationary (p_current/Left) (rf in lf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_next, p_cur = P_k_current, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, 
                                     g = g, glb = glb, gub = gub)
            #Relative Swing Foot Location - Stationary (p_current/Left) in Landing (p_next/Right) (lf in rf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_current, p_cur = P_k_next, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, 
                                     g = g, glb = glb, gub = gub)

            #------
            #Case 2
            #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left
            SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
            #Left foot is landing (p_next), Right Foot is stationary (p_current)
            #Relative Swing Foot Location - Landing (p_next/Left) in stationary (p_current/Right) (lf in rf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_next, p_cur = P_k_current, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, 
                                     g = g, glb = glb, gub = gub)
            #Relative Swing Foot Location - Stationary (p_current/Right) in Landing (p_next/Left) (rf in lf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_current, p_cur = P_k_next, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, 
                                     g = g, glb = glb, gub = gub)

        elif step_cnt%2 == 1: #odd number steps
            #-------
            #Case 1
            #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left
            SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
            #Left Foot is landing (p_next), Right Foot is stationary (p_current)
            #Relative Swing Foot Location - Landing (p_next/Left) in stationary (p_current/Right) (lf in rf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_next, p_cur = P_k_current, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, 
                                     g = g, glb = glb, gub = gub)
            #Relative Swing Foot Location - Stationary (p_current/Right) in Landing (p_next/Left) (rf in lf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_current, p_cur = P_k_next, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, 
                                     g = g, glb = glb, gub = gub)

            #-------
            #Case 2
            #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right
            SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
            #Right Foot is landing (p_next), Left foot is stationary (p_current)
            #Relative Swing Foot Location - Landing (p_next/Right) in Stationary (p_current/Left) (rf in lf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_next, p_cur = P_k_current, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, 
                                     g = g, glb = glb, gub = gub)
            #Relative Swing Foot Location - Stationary (p_current/Left) in Landing (p_next/Right) (lf in rf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_current, p_cur = P_k_next, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, 
                                     g = g, glb = glb, gub = gub)

    #-----------------
    #FootStep Location Constraint (On the Patch) -> Only One Step
    #P3----------------P1
    #|                  |
    #|                  |
    #|                  |
    #P4----------------P2

    for PatchNum in range(Nsteps): #No need to consider p_init, as they are constrained by the first level
        #Get Footstep Vector
        P_vector = ca.vertcat(px[PatchNum],py[PatchNum],pz[PatchNum])

        #Get Half Space Representation 
        #NOTE: In the second level, the terrain patch start from the second patch, indexed as 1
        SurfParaTemp = SurfParas[20+PatchNum*20:19+(PatchNum+1)*20+1]
        #print(SurfParaTemp)
        SurfK = SurfParaTemp[0:11+1]
        SurfK = ca.reshape(SurfK,3,4)
        SurfK = SurfK.T #NOTE: This process is due to casadi naming convention to have first row to be x1,x2,x3
        SurfE = SurfParaTemp[11+1:11+3+1]
        Surfk = SurfParaTemp[14+1:14+4+1]
        Surfe = SurfParaTemp[-1]

        #Terrain Tangent and Norms
        #In the second level, when getting the tangets, we need to have PatchNum/StepCnt + 1, 
        #As the second level counts the second step (enumerated as step 1 in entire horizon) as step 0
        P_vector_TangentX = SurfTangentsX[(PatchNum+1)*3:(PatchNum+1)*3+3]
        P_vector_TangentY = SurfTangentsY[(PatchNum+1)*3:(PatchNum+1)*3+3]

        g, glb, gub = Stay_on_Surf(P = P_vector, P_TangentX = P_vector_TangentX, P_TangentY = P_vector_TangentY, 
                                   ineq_K = SurfK, ineq_k = Surfk, eq_E = SurfE, eq_e = Surfe, g = g, glb = glb, gub = gub)

    # #-----------------------------------
    # #Switching Time Constraint
    # for phase_cnt in range(Nphase):
    #     if GaitPattern[phase_cnt] == 'InitialDouble':
    #         if phase_cnt == 0:
    #             g.append(Ts[phase_cnt] - 0)
    #             glb.append(np.array([0.1])) #old 0.1-0.3
    #             gub.append(np.array([0.3]))
    #         else:
    #             g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
    #             glb.append(np.array([0.1]))
    #             gub.append(np.array([0.3]))
    #     elif GaitPattern[phase_cnt] == 'Swing':
    #         g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
    #         glb.append(np.array([0.8])) #0.8-1.2
    #         gub.append(np.array([1.2]))
    #     elif GaitPattern[phase_cnt] == 'DoubleSupport':
    #         g.append(Ts[phase_cnt]-Ts[phase_cnt-1]) #0.1-0.9
    #         glb.append(np.array([0.1]))
    #         gub.append(np.array([0.3])) #old - 0.1-0.3
    #     else:
    #         raise Exception("Unknown Phase Name")

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Variable Index - !!!This is the pure Index, when try to get the array using other routines, we need to add "+1" at the last index due to Python indexing conventions
    #First set of variables start counting from 0, The enumeration of the last knot is N_K-1
    x_index = (0,N_K-1);                                      y_index = (x_index[1]+1,x_index[1]+N_K);                 z_index = (y_index[1]+1,y_index[1]+N_K)
    xdot_index = (z_index[1]+1,z_index[1]+N_K);               ydot_index = (xdot_index[1]+1,xdot_index[1]+N_K);        zdot_index = (ydot_index[1]+1,ydot_index[1]+N_K)
    FL1x_index = (zdot_index[1]+1,zdot_index[1]+N_K);         FL1y_index = (FL1x_index[1]+1,FL1x_index[1]+N_K);        FL1z_index = (FL1y_index[1]+1,FL1y_index[1]+N_K)
    FL2x_index = (FL1z_index[1]+1,FL1z_index[1]+N_K);         FL2y_index = (FL2x_index[1]+1,FL2x_index[1]+N_K);        FL2z_index = (FL2y_index[1]+1,FL2y_index[1]+N_K)
    FL3x_index = (FL2z_index[1]+1,FL2z_index[1]+N_K);         FL3y_index = (FL3x_index[1]+1,FL3x_index[1]+N_K);        FL3z_index = (FL3y_index[1]+1,FL3y_index[1]+N_K)
    FL4x_index = (FL3z_index[1]+1,FL3z_index[1]+N_K);         FL4y_index = (FL4x_index[1]+1,FL4x_index[1]+N_K);        FL4z_index = (FL4y_index[1]+1,FL4y_index[1]+N_K)
    FR1x_index = (FL4z_index[1]+1,FL4z_index[1]+N_K);         FR1y_index = (FR1x_index[1]+1,FR1x_index[1]+N_K);        FR1z_index = (FR1y_index[1]+1,FR1y_index[1]+N_K)
    FR2x_index = (FR1z_index[1]+1,FR1z_index[1]+N_K);         FR2y_index = (FR2x_index[1]+1,FR2x_index[1]+N_K);        FR2z_index = (FR2y_index[1]+1,FR2y_index[1]+N_K)
    FR3x_index = (FR2z_index[1]+1,FR2z_index[1]+N_K);         FR3y_index = (FR3x_index[1]+1,FR3x_index[1]+N_K);        FR3z_index = (FR3y_index[1]+1,FR3y_index[1]+N_K)
    FR4x_index = (FR3z_index[1]+1,FR3z_index[1]+N_K);         FR4y_index = (FR4x_index[1]+1,FR4x_index[1]+N_K);        FR4z_index = (FR4y_index[1]+1,FR4y_index[1]+N_K)
    px_init_index = (FR4z_index[1]+1,FR4z_index[1]+1);        py_init_index = (px_init_index[1]+1,px_init_index[1]+1); pz_init_index = (py_init_index[1]+1,py_init_index[1]+1)
    px_index = (pz_init_index[1]+1,pz_init_index[1]+Nsteps);  py_index = (px_index[1]+1,px_index[1]+Nsteps);           pz_index = (py_index[1]+1,py_index[1]+Nsteps)
    Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)
    
    cL1x_p_index = (Ts_index[1]+1,Ts_index[1]+N_K);           cL1x_q_index = (cL1x_p_index[1]+1,cL1x_p_index[1]+N_K)
    cL1y_p_index = (cL1x_q_index[1]+1,cL1x_q_index[1]+N_K);   cL1y_q_index = (cL1y_p_index[1]+1,cL1y_p_index[1]+N_K)
    cL1z_p_index = (cL1y_q_index[1]+1,cL1y_q_index[1]+N_K);   cL1z_q_index = (cL1z_p_index[1]+1,cL1z_p_index[1]+N_K)
    
    cL2x_p_index = (cL1z_q_index[1]+1,cL1z_q_index[1]+N_K);   cL2x_q_index = (cL2x_p_index[1]+1,cL2x_p_index[1]+N_K)
    cL2y_p_index = (cL2x_q_index[1]+1,cL2x_q_index[1]+N_K);   cL2y_q_index = (cL2y_p_index[1]+1,cL2y_p_index[1]+N_K)
    cL2z_p_index = (cL2y_q_index[1]+1,cL2y_q_index[1]+N_K);   cL2z_q_index = (cL2z_p_index[1]+1,cL2z_p_index[1]+N_K)
    
    cL3x_p_index = (cL2z_q_index[1]+1,cL2z_q_index[1]+N_K);   cL3x_q_index = (cL3x_p_index[1]+1,cL3x_p_index[1]+N_K)
    cL3y_p_index = (cL3x_q_index[1]+1,cL3x_q_index[1]+N_K);   cL3y_q_index = (cL3y_p_index[1]+1,cL3y_p_index[1]+N_K)
    cL3z_p_index = (cL3y_q_index[1]+1,cL3y_q_index[1]+N_K);   cL3z_q_index = (cL3z_p_index[1]+1,cL3z_p_index[1]+N_K)
    
    cL4x_p_index = (cL3z_q_index[1]+1,cL3z_q_index[1]+N_K);   cL4x_q_index = (cL4x_p_index[1]+1,cL4x_p_index[1]+N_K)
    cL4y_p_index = (cL4x_q_index[1]+1,cL4x_q_index[1]+N_K);   cL4y_q_index = (cL4y_p_index[1]+1,cL4y_p_index[1]+N_K)
    cL4z_p_index = (cL4y_q_index[1]+1,cL4y_q_index[1]+N_K);   cL4z_q_index = (cL4z_p_index[1]+1,cL4z_p_index[1]+N_K)
    
    cR1x_p_index = (cL4z_q_index[1]+1,cL4z_q_index[1]+N_K);   cR1x_q_index = (cR1x_p_index[1]+1,cR1x_p_index[1]+N_K)
    cR1y_p_index = (cR1x_q_index[1]+1,cR1x_q_index[1]+N_K);   cR1y_q_index = (cR1y_p_index[1]+1,cR1y_p_index[1]+N_K)
    cR1z_p_index = (cR1y_q_index[1]+1,cR1y_q_index[1]+N_K);   cR1z_q_index = (cR1z_p_index[1]+1,cR1z_p_index[1]+N_K)
    
    cR2x_p_index = (cR1z_q_index[1]+1,cR1z_q_index[1]+N_K);   cR2x_q_index = (cR2x_p_index[1]+1,cR2x_p_index[1]+N_K)
    cR2y_p_index = (cR2x_q_index[1]+1,cR2x_q_index[1]+N_K);   cR2y_q_index = (cR2y_p_index[1]+1,cR2y_p_index[1]+N_K)
    cR2z_p_index = (cR2y_q_index[1]+1,cR2y_q_index[1]+N_K);   cR2z_q_index = (cR2z_p_index[1]+1,cR2z_p_index[1]+N_K)
    
    cR3x_p_index = (cR2z_q_index[1]+1,cR2z_q_index[1]+N_K);   cR3x_q_index = (cR3x_p_index[1]+1,cR3x_p_index[1]+N_K)
    cR3y_p_index = (cR3x_q_index[1]+1,cR3x_q_index[1]+N_K);   cR3y_q_index = (cR3y_p_index[1]+1,cR3y_p_index[1]+N_K)
    cR3z_p_index = (cR3y_q_index[1]+1,cR3y_q_index[1]+N_K);   cR3z_q_index = (cR3z_p_index[1]+1,cR3z_p_index[1]+N_K)
    
    cR4x_p_index = (cR3z_q_index[1]+1,cR3z_q_index[1]+N_K);   cR4x_q_index = (cR4x_p_index[1]+1,cR4x_p_index[1]+N_K)
    cR4y_p_index = (cR4x_q_index[1]+1,cR4x_q_index[1]+N_K);   cR4y_q_index = (cR4y_p_index[1]+1,cR4y_p_index[1]+N_K)
    cR4z_p_index = (cR4y_q_index[1]+1,cR4y_q_index[1]+N_K);   cR4z_q_index = (cR4z_p_index[1]+1,cR4z_p_index[1]+N_K)

    var_index = {"x":x_index,        "y":y_index,         "z":z_index,         "xdot":xdot_index,     "ydot":ydot_index,     "zdot":zdot_index,
                 "FL1x":FL1x_index,  "FL1y":FL1y_index,   "FL1z":FL1z_index,   
                 "FL2x":FL2x_index,  "FL2y":FL2y_index,   "FL2z":FL2z_index,
                 "FL3x":FL3x_index,  "FL3y":FL3y_index,   "FL3z":FL3z_index,
                 "FL4x":FL4x_index,  "FL4y":FL4y_index,   "FL4z":FL4z_index,
                 "FR1x":FR1x_index,  "FR1y":FR1y_index,   "FR1z":FR1z_index,
                 "FR2x":FR2x_index,  "FR2y":FR2y_index,   "FR2z":FR2z_index,
                 "FR3x":FR3x_index,  "FR3y":FR3y_index,   "FR3z":FR3z_index,
                 "FR4x":FR4x_index,  "FR4y":FR4y_index,   "FR4z":FR4z_index,
                 "px_init":px_init_index,   "py_init":py_init_index,   "pz_init":pz_init_index,
                 "px":px_index,   "py":py_index,   "pz":pz_index,
                 "Ts":Ts_index,
                 "cL1x_p":cL1x_p_index,  "cL1x_q":cL1x_q_index,  "cL1y_p":cL1y_p_index,  "cL1y_q":cL1y_q_index,  "cL1z_p":cL1z_p_index,  "cL1z_q":cL1z_q_index,
                 "cL2x_p":cL2x_p_index,  "cL2x_q":cL2x_q_index,  "cL2y_p":cL2y_p_index,  "cL2y_q":cL2y_q_index,  "cL2z_p":cL2z_p_index,  "cL2z_q":cL2z_q_index,
                 "cL3x_p":cL3x_p_index,  "cL3x_q":cL3x_q_index,  "cL3y_p":cL3y_p_index,  "cL3y_q":cL3y_q_index,  "cL3z_p":cL3z_p_index,  "cL3z_q":cL3z_q_index,
                 "cL4x_p":cL4x_p_index,  "cL4x_q":cL4x_q_index,  "cL4y_p":cL4y_p_index,  "cL4y_q":cL4y_q_index,  "cL4z_p":cL4z_p_index,  "cL4z_q":cL4z_q_index,
                 "cR1x_p":cR1x_p_index,  "cR1x_q":cR1x_q_index,  "cR1y_p":cR1y_p_index,  "cR1y_q":cR1y_q_index,  "cR1z_p":cR1z_p_index,  "cR1z_q":cR1z_q_index,  
                 "cR2x_p":cR2x_p_index,  "cR2x_q":cR2x_q_index,  "cR2y_p":cR2y_p_index,  "cR2y_q":cR2y_q_index,  "cR2z_p":cR2z_p_index,  "cR2z_q":cR2z_q_index,
                 "cR3x_p":cR3x_p_index,  "cR3x_q":cR3x_q_index,  "cR3y_p":cR3y_p_index,  "cR3y_q":cR3y_q_index,  "cR3z_p":cR3z_p_index,  "cR3z_q":cR3z_q_index,
                 "cR4x_p":cR4x_p_index,  "cR4x_q":cR4x_q_index,  "cR4y_p":cR4y_p_index,  "cR4y_q":cR4y_q_index,  "cR4z_p":cR4z_p_index,  "cR4z_q":cR4z_q_index,
    }

    return DecisionVars, DecisionVars_lb, DecisionVars_ub, J, g, glb, gub, var_index

#NOTE: Ponton Methods do not have rotated kinematics polytopes
def Ponton_SinglePoint(m = 100.0, Nk_Local = 7, Nsteps = 1, ParameterList = None, PontonTerm_bounds = 0.55):

    #-------------------------------------------
    #Define Constant Parameters
    #   Gravitational Acceleration
    G = 9.80665 #kg/m^2
    #   Friciton Coefficient 
    miu = 0.3
    #   Force Limits
    F_bound = 400.0 * 4.0
    Fxlb = -F_bound;   Fxub = F_bound
    Fylb = -F_bound;   Fyub = F_bound
    Fzlb = -F_bound;   Fzub = F_bound
    #   Angular Momentum Bounds
    #L_bound = 2.5;      Llb = -L_bound;         Lub = L_bound
    #Ldot_bound = 3.5;   Ldotlb = -Ldot_bound;   Ldotub = Ldot_bound
    #   Bounds on CoM Height
    z_lowest = -5.0
    z_highest = 5.0
    #   CoM Height with respect to Footstep Location
    CoM_z_to_Foot_min = 0.65
    CoM_z_to_Foot_max = 0.75
    #   Ponton Term Bounds
    p_lb = -PontonTerm_bounds;     p_ub = PontonTerm_bounds
    q_lb = -PontonTerm_bounds;     q_ub = PontonTerm_bounds
    #   Leg Length (Normalisation) for Ponton
    max_leg_length = 1.5 #can be 1.45
    #---------------------------------------------
    #Define Parameters
    #   Gait Pattern, Each action is followed up by a double support phase
    GaitPattern = ["InitialDouble","Swing","DoubleSupport"] + ["InitialDouble", "Swing","DoubleSupport"]*(Nsteps-1) #,'RightSupport','DoubleSupport','LeftSupport','DoubleSupport'
    #   Number of Phases
    Nphase = len(GaitPattern)
    #   Compute Number of Total knots/ticks, but the enumeration start from 0 to N_K-1
    N_K = Nk_Local*Nphase + 1 #+1 the last knots to finalize the plan
    #   Phase Duration Vector; NOTE: Mannually defined
    #PhaseDurationVec = [0.3, 0.8, 0.3]*(Nsteps) + [0.3, 0.8, 0.3]*(Nsteps-1)
    PhaseDurationVec = [0.2, 0.5, 0.2]*(Nsteps) + [0.2, 0.5, 0.2]*(Nsteps-1)

    #-----------------------------------------------------------------------------------------------------------------------
    #Load kinematics Polytope
    # #   Not local than server
    # kinefilepath = "/home/jiayu/Desktop/multicontact_learning_local_objectives/python/kinematics_polytope/kinematics_constraints.p"
    # if os.path.isfile(kinefilepath) == False:
    #     kinefilepath = "/afs/inf.ed.ac.uk/user/s15/s1545529/Desktop/multicontact_learning_local_objectives/python/kinematics_polytope/kinematics_constraints.p"

    # with open(kinefilepath, 'rb') as f:
    #     kinematics_constraints= pickle.load(f)
    
    # #CoM Polytopes
    # K_CoM_Right = kinematics_constraints["K_CoM_in_Right_Contact"];     k_CoM_Right = kinematics_constraints["k_CoM_in_Right_Contact"]
    # K_CoM_Left  = kinematics_constraints["K_CoM_in_Left_Contact"];      k_CoM_Left  = kinematics_constraints["k_CoM_in_Left_Contact"]

    # #Relative Footstep constraints
    # Q_rf_in_lf = kinematics_constraints["Q_Right_Contact_in_Left_Contact"];    q_rf_in_lf = kinematics_constraints["q_Right_Contact_in_Left_Contact"]
    # Q_lf_in_rf = kinematics_constraints["Q_Left_Contact_in_Right_Contact"];    q_lf_in_rf = kinematics_constraints["q_Left_Contact_in_Right_Contact"]

    #------------------------------------------

    #Get Kinematics Constraint for Talos
    #CoM kinematics constraint, give homogenous transformaiton (the last column seems like dont make a diff)
    K_CoM_Right,k_CoM_Right = right_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    K_CoM_Left,k_CoM_Left = left_foot_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    #Another way
    #kinematicConstraints = genKinematicConstraints(left_foot_constraints,right_foot_constraints)
    #K_CoM_Left = kinematicConstraints[0][0];   k_CoM_Left = kinematicConstraints[0][1]
    #K_CoM_Right = kinematicConstraints[1][0];  k_CoM_Right = kinematicConstraints[1][1]
    
    #Relative Foot Constraint matrices
    #Relative foot constraint, give homogenous transformation (the last column seems like dont make a diff)
    Q_rf_in_lf,q_rf_in_lf = right_foot_in_lf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    Q_lf_in_rf,q_lf_in_rf = left_foot_in_rf_frame_constraints(np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]]))
    #Another way
    #relativeConstraints = genFootRelativeConstraints(right_foot_in_lf_frame_constraints,left_foot_in_rf_frame_constraints)
    #Q_rf_in_lf = relativeConstraints[0][0];   q_rf_in_lf = relativeConstraints[0][1] #named lf in rf, but representing rf in lf
    #Q_lf_in_rf = relativeConstraints[1][0];   q_lf_in_rf = relativeConstraints[1][1] #named rf in lf, but representing lf in rf

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Casadi Parameters
    #Flags for Swing Legs (Defined as Parameters)
    ParaLeftSwingFlag = ParameterList["LeftSwingFlag"]
    ParaRightSwingFlag = ParameterList["RightSwingFlag"]

    #Initial Left Foot Contact Location
    PLx_init = ParameterList["PLx_init"];   PLy_init = ParameterList["PLy_init"];   PLz_init = ParameterList["PLz_init"]
    PL_init = ca.vertcat(PLx_init,PLy_init,PLz_init)
    #Initial Right Foot Contact Location
    PRx_init = ParameterList["PRx_init"];   PRy_init = ParameterList["PRy_init"];   PRz_init = ParameterList["PRz_init"]
    PR_init = ca.vertcat(PRx_init,PRy_init,PRz_init)

    #Surface Patches
    SurfParas = ParameterList["SurfParas"]

    #Tangents and Norms
    #Initial Contact Norm and Tangents
    PL_init_Norm = ParameterList["PL_init_Norm"];   PL_init_TangentX = ParameterList["PL_init_TangentX"];   PL_init_TangentY = ParameterList["PL_init_TangentY"]
    PR_init_Norm = ParameterList["PR_init_Norm"];   PR_init_TangentX = ParameterList["PR_init_TangentX"];   PR_init_TangentY = ParameterList["PR_init_TangentY"]

    #Future Contact Norm and Tangents
    SurfNorms = ParameterList["SurfNorms"]                
    SurfTangentsX = ParameterList["SurfTangentsX"]
    SurfTangentsY = ParameterList["SurfTangentsY"]

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Variables and Bounds, Parameters
    #   CoM Position x, y, z
    x = ca.SX.sym('x',N_K);   x_lb = np.array([[0.0]*(x.shape[0]*x.shape[1])]);         x_ub = np.array([[50.0]*(x.shape[0]*x.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    y = ca.SX.sym('y',N_K);   y_lb = np.array([[-1.0]*(y.shape[0]*y.shape[1])]);        y_ub = np.array([[1.0]*(y.shape[0]*y.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    z = ca.SX.sym('z',N_K);   z_lb = np.array([[z_lowest]*(z.shape[0]*z.shape[1])]);  z_ub = np.array([[z_highest]*(z.shape[0]*z.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #   CoM Velocity x, y, z
    xdot = ca.SX.sym('xdot',N_K);   xdot_lb = np.array([[-1.5]*(xdot.shape[0]*xdot.shape[1])]);   xdot_ub = np.array([[1.5]*(xdot.shape[0]*xdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    ydot = ca.SX.sym('ydot',N_K);   ydot_lb = np.array([[-1.5]*(ydot.shape[0]*ydot.shape[1])]);   ydot_ub = np.array([[1.5]*(ydot.shape[0]*ydot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    zdot = ca.SX.sym('zdot',N_K);   zdot_lb = np.array([[-1.5]*(zdot.shape[0]*zdot.shape[1])]);   zdot_ub = np.array([[1.5]*(zdot.shape[0]*zdot.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Left Foot Forces x, y, z
    FLx = ca.SX.sym('FLx',N_K);   FLx_lb = np.array([[Fxlb]*(FLx.shape[0]*FLx.shape[1])]);   FLx_ub = np.array([[Fxub]*(FLx.shape[0]*FLx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FLy = ca.SX.sym('FLy',N_K);   FLy_lb = np.array([[Fylb]*(FLy.shape[0]*FLy.shape[1])]);   FLy_ub = np.array([[Fyub]*(FLy.shape[0]*FLy.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FLz = ca.SX.sym('FLz',N_K);   FLz_lb = np.array([[Fzlb]*(FLz.shape[0]*FLz.shape[1])]);   FLz_ub = np.array([[Fzub]*(FLz.shape[0]*FLz.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    #Right Contact Force x, y, z
    FRx = ca.SX.sym('FRx',N_K);   FRx_lb = np.array([[Fxlb]*(FRx.shape[0]*FRx.shape[1])]);   FRx_ub = np.array([[Fxub]*(FRx.shape[0]*FRx.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FRy = ca.SX.sym('FRy',N_K);   FRy_lb = np.array([[Fylb]*(FRy.shape[0]*FRy.shape[1])]);   FRy_ub = np.array([[Fyub]*(FRy.shape[0]*FRy.shape[1])]) #particular way of generating lists in python, [value]*number of elements
    FRz = ca.SX.sym('FRz',N_K);   FRz_lb = np.array([[Fzlb]*(FRz.shape[0]*FRz.shape[1])]);   FRz_ub = np.array([[Fzub]*(FRz.shape[0]*FRz.shape[1])]) #particular way of generating lists in python, [value]*number of elements

    #Initial Contact Location (First step made in the first level), need to connect to the first level landing location 
    #   Px, Py, Pz
    px_init = ca.SX.sym('px_init');   px_init_lb = np.array([-1.0]);   px_init_ub = np.array([50.0])
    py_init = ca.SX.sym('py_init');   py_init_lb = np.array([-2.0]);   py_init_ub = np.array([2.0])
    pz_init = ca.SX.sym('pz_init');   pz_init_lb = np.array([-5.0]);   pz_init_ub = np.array([5.0])
    
    #Contact Location Sequence
    px = [];   px_lb = [];   px_ub = []
    py = [];   py_lb = [];   py_ub = []
    pz = [];   pz_lb = [];   pz_ub = []
    for stepIdx in range(Nsteps):
        #Nsteps: Number of steps in the second level, = Total Number of Steps of the Entire Lookahead Horizon - 1
        #Therefore the enumeration of contact location sequences start counting from 1 (to be aligned with step number in the entire horizon)
        pxtemp = ca.SX.sym('px'+str(stepIdx + 1));   px.append(pxtemp);   px_lb.append(np.array([-1.0]));   px_ub.append(np.array([50.0]))
        pytemp = ca.SX.sym('py'+str(stepIdx + 1));   py.append(pytemp);   py_lb.append(np.array([-2.0]));   py_ub.append(np.array([2.0]))
        pztemp = ca.SX.sym('pz'+str(stepIdx + 1));   pz.append(pztemp);   pz_lb.append(np.array([-5.0]));   pz_ub.append(np.array([5.0]))

    #Switching Time Vector
    Ts = [];   Ts_lb = [];   Ts_ub = []
    for n_phase in range(Nphase):
        #Ts start counting from 1, Ts0 = 0
        Tstemp = ca.SX.sym('Ts'+str(n_phase+1));   Ts.append(Tstemp);   Ts_lb.append(np.array([0.05]));   Ts_ub.append(np.array([3.0*(Nphase+1)]))    
    
    #Ponton Terms (p and q for x, y, z angular momentum rate)
    #Left Foot (p and q for x, y, z)
    #   x-axis
    cLx_p = ca.SX.sym('cLx_p',N_K);  cLx_p_lb = np.array([[p_lb]*(cLx_p.shape[0]*cLx_p.shape[1])]);  cLx_p_ub = np.array([[p_ub]*(cLx_p.shape[0]*cLx_p.shape[1])])
    cLx_q = ca.SX.sym('cLx_q',N_K);  cLx_q_lb = np.array([[q_lb]*(cLx_q.shape[0]*cLx_q.shape[1])]);  cLx_q_ub = np.array([[q_ub]*(cLx_q.shape[0]*cLx_q.shape[1])])
    #   y-axis
    cLy_p = ca.SX.sym('cLy_p',N_K);  cLy_p_lb = np.array([[p_lb]*(cLy_p.shape[0]*cLy_p.shape[1])]);  cLy_p_ub = np.array([[p_ub]*(cLy_p.shape[0]*cLy_p.shape[1])])
    cLy_q = ca.SX.sym('cLy_q',N_K);  cLy_q_lb = np.array([[q_lb]*(cLy_q.shape[0]*cLy_q.shape[1])]);  cLy_q_ub = np.array([[q_ub]*(cLy_q.shape[0]*cLy_q.shape[1])])
    #   z-axis
    cLz_p = ca.SX.sym('cLz_p',N_K);  cLz_p_lb = np.array([[p_lb]*(cLz_p.shape[0]*cLz_p.shape[1])]);  cLz_p_ub = np.array([[p_ub]*(cLz_p.shape[0]*cLz_p.shape[1])])
    cLz_q = ca.SX.sym('cLz_q',N_K);  cLz_q_lb = np.array([[q_lb]*(cLz_q.shape[0]*cLz_q.shape[1])]);  cLz_q_ub = np.array([[q_ub]*(cLz_q.shape[0]*cLz_q.shape[1])])
    #Right Foot (p and q for x, y, z)
    #   x-axis
    cRx_p = ca.SX.sym('cRx_p',N_K);  cRx_p_lb = np.array([[p_lb]*(cRx_p.shape[0]*cRx_p.shape[1])]);  cRx_p_ub = np.array([[p_ub]*(cRx_p.shape[0]*cRx_p.shape[1])])
    cRx_q = ca.SX.sym('cRx_q',N_K);  cRx_q_lb = np.array([[q_lb]*(cRx_q.shape[0]*cRx_q.shape[1])]);  cRx_q_ub = np.array([[q_ub]*(cRx_q.shape[0]*cRx_q.shape[1])])
    #   y-axis
    cRy_p = ca.SX.sym('cRy_p',N_K);  cRy_p_lb = np.array([[p_lb]*(cRy_p.shape[0]*cRy_p.shape[1])]);  cRy_p_ub = np.array([[p_ub]*(cRy_p.shape[0]*cRy_p.shape[1])])
    cRy_q = ca.SX.sym('cRy_q',N_K);  cRy_q_lb = np.array([[q_lb]*(cRy_q.shape[0]*cRy_q.shape[1])]);  cRy_q_ub = np.array([[q_ub]*(cRy_q.shape[0]*cRy_q.shape[1])])
    #   z-axis
    cRz_p = ca.SX.sym('cRz_p',N_K);  cRz_p_lb = np.array([[p_lb]*(cRz_p.shape[0]*cRz_p.shape[1])]);  cRz_p_ub = np.array([[p_ub]*(cRz_p.shape[0]*cRz_p.shape[1])])
    cRz_q = ca.SX.sym('cRz_q',N_K);  cRz_q_lb = np.array([[q_lb]*(cRz_q.shape[0]*cRz_q.shape[1])]);  cRz_q_ub = np.array([[q_ub]*(cRz_q.shape[0]*cRz_q.shape[1])])    

    #   Collect all Decision Variables
    DecisionVars = ca.vertcat(x,        y,        z,
                              xdot,     ydot,     zdot,
                              FLx,      FLy,      FLz,     
                              FRx,      FRy,      FRz,     
                              px_init,  py_init,  pz_init,
                              *px,      *py,      *pz,
                              *Ts,
                              cLx_p,    cLx_q,    cLy_p,    cLy_q,    cLz_p,    cLz_q,
                              cRx_p,    cRx_q,    cRy_p,    cRy_q,    cRz_p,    cRz_q)
    #print(DecisionVars)
    DecisionVarsShape = DecisionVars.shape

    #   Collect all lower bound and upper bound
    DecisionVars_lb = (x_lb,       y_lb,       z_lb,
                       xdot_lb,    ydot_lb,    zdot_lb,
                       FLx_lb,     FLy_lb,     FLz_lb,
                       FRx_lb,     FRy_lb,     FRz_lb,
                       px_init_lb, py_init_lb, pz_init_lb,
                       px_lb,      py_lb,      pz_lb,
                       Ts_lb,
                       cLx_p_lb,   cLx_q_lb,   cLy_p_lb,   cLy_q_lb,   cLz_p_lb,   cLz_q_lb,
                       cRx_p_lb,   cRx_q_lb,   cRy_p_lb,   cRy_q_lb,   cRz_p_lb,   cRz_q_lb)
    DecisionVars_lb = np.concatenate(DecisionVars_lb,axis=None)

    DecisionVars_ub = (x_ub,       y_ub,       z_ub,
                       xdot_ub,    ydot_ub,    zdot_ub,
                       FLx_ub,     FLy_ub,     FLz_ub, 
                       FRx_ub,     FRy_ub,     FRz_ub, 
                       px_init_ub, py_init_ub, pz_init_ub,
                       px_ub,      py_ub,      pz_ub,
                       Ts_ub,
                       cLx_p_ub,   cLx_q_ub,   cLy_p_ub,   cLy_q_ub,   cLz_p_ub,   cLz_q_ub,
                       cRx_p_ub,   cRx_q_ub,   cRy_p_ub,   cRy_q_ub,   cRz_p_ub,   cRz_q_ub)
    DecisionVars_ub = np.concatenate(DecisionVars_ub,axis=None)

    #-----------------------------------------------------------------------------------------------------------------------
    #Define Constrains and Running Cost
    g = [];   glb = [];   gub = []
    J = 0

    #Constraints for all knots
    #Loop over all Phases (Knots)
    for Nph in range(Nphase):
        #Decide Number of Knots
        if Nph == Nphase-1:  #The last Knot belongs to the Last Phase
            Nk_ThisPhase = Nk_Local+1
        else:
            Nk_ThisPhase = Nk_Local  
        
        #Decide Time Step (Fixed)
        h = (PhaseDurationVec[Nph])/Nk_Local

        for Local_k_Count in range(Nk_ThisPhase):
            #Get knot number across the entire time line
            k = Nph*Nk_Local + Local_k_Count

            #------------------------------------------
            #Build useful vectors
            #   Forces
            FL_k = ca.vertcat(FLx[k],FLy[k],FLz[k])
            FR_k = ca.vertcat(FRx[k],FRy[k],FRz[k])
            #   CoM
            CoM_k = ca.vertcat(x[k],y[k],z[k])

            #-------------------------------------------
            #Phase dependent Constraints (CoM Kinematics and Angular Dynamics)
            #Get Step Counter
            StepCnt = Nph//3

            if GaitPattern[Nph]=='InitialDouble':
                #Special Case:
                if StepCnt == 0: #The first phase in the First STEP (Initial Double, need special care)
                    #initial support foot (the landing foot from the first phase)
                    p_init = ca.vertcat(px_init,py_init,pz_init)
                    p_init_TangentX = SurfTangentsX[0:3]
                    p_init_TangentY = SurfTangentsY[0:3]
                    p_init_Norm = SurfNorms[0:3]

                    #-----------
                    #Case 1
                    #If First Level Swing the Left, the the 0 phase (InitDouble) has p_init as the left support, PR_init as the right support
                    #(same as the double support phase of the first step)-> Left foot Moved (p_init), Right Foot stay stationary (PR_init)
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Kinematics Constraint
                    #CoM in Left (p_init)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_init, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM in Right (PR_init)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PR_init, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)

                    #CoM Height Constraint (Left p_init foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #CoM Height Constraint (Right foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PR_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Center",   P = p_init,   P_TangentX = p_init_TangentX,   P_TangentY = p_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FL_k,             f_length = F_bound,
                                                               x_p_bar = cLx_p[k], x_q_bar = cLx_q[k], y_p_bar = cLy_p[k], y_q_bar = cLy_q[k], 
                                                               z_p_bar = cLz_p[k], z_q_bar = cLz_q[k], g = g, glb = glb, gub = gub)
                        #--------------------------
                        #Right Foot
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Center",   P = PR_init,   P_TangentX = PR_init_TangentX,   P_TangentY = PR_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FR_k,             f_length = F_bound,
                                                               x_p_bar = cRx_p[k], x_q_bar = cRx_q[k], y_p_bar = cRy_p[k], y_q_bar = cRy_q[k], 
                                                               z_p_bar = cRz_p[k], z_q_bar = cRz_q[k], g = g, glb = glb, gub = gub)

                    #Unilateral Constraint
                    #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the unilateral constraint on p_init
                    #Left Foot force (p_init)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the PR_init
                    #Right Foot (PR_init)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainNorm = PR_init_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #If the first level swing the Left foot first, then the Left foot is the landing foot (p_init), Left foot obey the friction cone constraint on p_init
                    #Left Foot (p_init)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the PR_init
                    #Right Foot (PR_init)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainTangentX = PR_init_TangentX, TerrainTangentY = PR_init_TangentY, TerrainNorm = PR_init_Norm, miu = miu, g = g, glb = glb, gub = gub)

                    #--------------------
                    #Case 2
                    #If First Level Swing the Right, the the 0 phase (InitDouble) has p_init as the Right support, PL_init as the Left support
                    #(same as the double support phase of the first step)-> Right foot Moved (p_init), Left Foot stay stationary (PL_init)
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Kinematics Constraint
                    #CoM in Left (PL_init)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PL_init, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM in Right (p_init)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_init, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                
                    #CoM Height Constraint (Left PL_init foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = PL_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #CoM Height Constraint (Right p_init foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_init, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics
                    if k<N_K-1:
                        #------------------------------
                        #Left Foot
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Center",   P = PL_init,   P_TangentX = PL_init_TangentX,   P_TangentY = PL_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FL_k,             f_length = F_bound,
                                                               x_p_bar = cLx_p[k], x_q_bar = cLx_q[k], y_p_bar = cLy_p[k], y_q_bar = cLy_q[k], 
                                                               z_p_bar = cLz_p[k], z_q_bar = cLz_q[k], g = g, glb = glb, gub = gub)

                        #--------------------------
                        #Right Foot
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                               P_name = "Center",   P = p_init,   P_TangentX = p_init_TangentX,   P_TangentY = p_init_TangentY,
                                                               CoM_k = CoM_k,         l_length = max_leg_length,
                                                               f = FR_k,             f_length = F_bound,
                                                               x_p_bar = cRx_p[k], x_q_bar = cRx_q[k], y_p_bar = cRy_p[k], y_q_bar = cRy_q[k], 
                                                               z_p_bar = cRz_p[k], z_q_bar = cRz_q[k], g = g, glb = glb, gub = gub)
                    #Unilateral Constraint
                    #Left Foot (PL_init)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainNorm = PL_init_Norm, g = g, glb = glb, gub = gub)
                    #Right Foot (p_init)
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainNorm = p_init_Norm, g = g, glb = glb, gub = gub)
                    
                    #Friction Cone Constraint
                    #Left Foot (PL_init)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainTangentX = PL_init_TangentX, TerrainTangentY = PL_init_TangentY, TerrainNorm = PL_init_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #Right Foot (p_init)
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainTangentX = p_init_TangentX, TerrainTangentY = p_init_TangentY, TerrainNorm = p_init_Norm, miu = miu, g = g, glb = glb, gub = gub)

                elif StepCnt > 0:#Other Cases
                    #Get contact location and Terrain Tangents and Norms
                    if StepCnt == 1: #Step 1 needs special care (NOTE: Step Count Start from 0)
                        #Previous Step
                        p_previous = ca.vertcat(px_init,py_init,pz_init)
                        p_previous_TangentX = SurfTangentsX[0:3];   p_previous_TangentY = SurfTangentsY[0:3]
                        p_previous_Norm = SurfNorms[0:3]

                        #Current Step
                        #In second level, Surfaces index is Step Vector Index (for px, py, pz, here is StepCnt-1) + 1
                        #For Intial Double Support, previous step is StepNum - 2, current step is StepNum - 1
                        #but index for the tangents and norm need to +1 as StepNum (in the second level) + 1 = StepNum in the entire horizon
                        p_current = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_current_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3];   p_current_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3] 
                        p_current_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    else: #Like Step 2, 3, 4 .....
                        #For Intial Double Support, previous step is StepNum - 2, current step is StepNum - 1
                        #but index for the tangents and norm need to +1 as StepNum (in the second level) + 1 = StepNum in the entire horizon
                        #For Initial Double Support, the contact config is the same as the double support phase of the previous step, where p_current is the landed foot
                        #p_previous is the non-moving foot
                        p_previous = ca.vertcat(px[StepCnt-2],py[StepCnt-2],pz[StepCnt-2])
                        p_previous_TangentX = SurfTangentsX[(StepCnt-1)*3:(StepCnt-1)*3+3];   p_previous_TangentY = SurfTangentsY[(StepCnt-1)*3:(StepCnt-1)*3+3]
                        p_previous_Norm = SurfNorms[(StepCnt-1)*3:(StepCnt-1)*3+3]

                        p_current = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                        p_current_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3];   p_current_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                        p_current_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    #Give Constraint according to even and odd steps
                    if StepCnt%2 == 0: #Even Numbers of Footsteps
                        #----------
                        #Case 1
                        #If the first level swing the Left, then the Even Number of Steps in the Intial Double support phase have p_current as Left Support (Landed), p_previous as Right Support (Stationary)
                        #(same as the double support phase of the first step)->Left foot Moved (p_current), Right Foot stay stationary p_previous)
                        SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                        #CoM in the Left (p_current)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                        #CoM in the Right (p_previous)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Left p_current foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Right p_previous foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #Angular Dynamics (Ponton)
                        if k<N_K-1: #double check the knot number is valid
                            #------------------------------
                            #Left Foot
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Center",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL_k,             f_length = F_bound,
                                                                x_p_bar = cLx_p[k], x_q_bar = cLx_q[k], y_p_bar = cLy_p[k], y_q_bar = cLy_q[k], 
                                                                z_p_bar = cLz_p[k], z_q_bar = cLz_q[k], g = g, glb = glb, gub = gub)
                            #--------------------------
                            #Right Foot
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Center",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR_k,             f_length = F_bound,
                                                                x_p_bar = cRx_p[k], x_q_bar = cRx_q[k], y_p_bar = cRy_p[k], y_q_bar = cRy_q[k], 
                                                                z_p_bar = cRz_p[k], z_q_bar = cRz_q[k], g = g, glb = glb, gub = gub)
                        #Unilateral Constraint
                        #Left Foot (p_current)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        #Right Foot (p_previous)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        #Friction Cone Constraint
                        #Left Foot (p_current)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        #Right Foot (p_previous)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        
                        #---------
                        #Case 2
                        #If the first level swing the Right, then the Even Number of Steps in the Intial Double support phase have p_current as Right Support (Landed), 
                        #(same as the double support phase of the first step) -> Right Moved (p_current), Left stationary (p_previous)
                        SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                        #CoM in the Left (p_previous)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                        #CoM in the Right (p_current)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Left p_previous foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Right p_current foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #Angular Dynamics (Ponton)
                        if k<N_K-1: #double check the knot number is valid
                            #------------------------------
                            #Left Foot
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Center",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL_k,             f_length = F_bound,
                                                                x_p_bar = cLx_p[k], x_q_bar = cLx_q[k], y_p_bar = cLy_p[k], y_q_bar = cLy_q[k], 
                                                                z_p_bar = cLz_p[k], z_q_bar = cLz_q[k], g = g, glb = glb, gub = gub)
                            #--------------------------
                            #Right Foot
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Center",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR_k,             f_length = F_bound,
                                                                x_p_bar = cRx_p[k], x_q_bar = cRx_q[k], y_p_bar = cRy_p[k], y_q_bar = cRy_q[k], 
                                                                z_p_bar = cRz_p[k], z_q_bar = cRz_q[k], g = g, glb = glb, gub = gub)

                        #Unilateral Constraint
                        #Left foot (p_previous)                         
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        #Right foot (p_current)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        #Friction Cone Constraint
                        #Left foot (p_previous)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        #Right Foot (p_current)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    elif StepCnt%2 == 1: #Odd Number of Steps
                        #---------
                        #Case 1
                        #If the first level swing the Left, then the Odd Number of Steps in the Intial Double support phase (Swing Right) have p_current as Right Support (Landed), p_previous as Left Support (Stationary)
                        #(same as the double support phase of the first step) -> Right Moved (p_current), Left Stay Fixed (p_previous)
                        SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                        #CoM in the Left (p_previous)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                        #CoM in the Right (p_current)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Left p_previous foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Right p_current foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #Angular Dynamics (Ponton)
                        if k<N_K-1: #double check the knot number is valid
                            #------------------------------
                            #Left Foot
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Center",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL_k,             f_length = F_bound,
                                                                x_p_bar = cLx_p[k], x_q_bar = cLx_q[k], y_p_bar = cLy_p[k], y_q_bar = cLy_q[k], 
                                                                z_p_bar = cLz_p[k], z_q_bar = cLz_q[k], g = g, glb = glb, gub = gub)
                            #--------------------------
                            #Right Foot
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Center",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR_k,             f_length = F_bound,
                                                                x_p_bar = cRx_p[k], x_q_bar = cRx_q[k], y_p_bar = cRy_p[k], y_q_bar = cRy_q[k], 
                                                                z_p_bar = cRz_p[k], z_q_bar = cRz_q[k], g = g, glb = glb, gub = gub)
                        #Unilateral Constraint
                        #Left foot (p_previous)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        #Right foot (p_current)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        #Friction Cone Constraint
                        #Left foot (p_previous)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        #Right Foot (p_current)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        
                        #-----
                        #Case 2
                        #If the first level swing the Right, then the Odd Number of Steps in the Intial Double support phase have p_current as Left Support (Landed), 
                        #(same as the double support phase of the first step) -> Left Moved (p_current), Right stationary (p_previous)
                        SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                        #CoM in the Left (p_current)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                        #CoM in the Right (p_previous)
                        g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Left p_current foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_current, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #   CoM Height Constraint (Right p_previous foot)
                        g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_previous, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                        #Angular Dynamics (Ponton)
                        if k<N_K-1: #double check the knot number is valid
                            #------------------------------
                            #Left Foot
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Center",   P = p_current,   P_TangentX = p_current_TangentX,   P_TangentY = p_current_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FL_k,             f_length = F_bound,
                                                                x_p_bar = cLx_p[k], x_q_bar = cLx_q[k], y_p_bar = cLy_p[k], y_q_bar = cLy_q[k], 
                                                                z_p_bar = cLz_p[k], z_q_bar = cLz_q[k], g = g, glb = glb, gub = gub)
                            #--------------------------
                            #Right Foot
                            g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                                P_name = "Center",   P = p_previous,   P_TangentX = p_previous_TangentX,   P_TangentY = p_previous_TangentY,
                                                                CoM_k = CoM_k,         l_length = max_leg_length,
                                                                f = FR_k,             f_length = F_bound,
                                                                x_p_bar = cRx_p[k], x_q_bar = cRx_q[k], y_p_bar = cRy_p[k], y_q_bar = cRy_q[k], 
                                                                z_p_bar = cRz_p[k], z_q_bar = cRz_q[k], g = g, glb = glb, gub = gub)
                        #Unilateral Constraint
                        #Left foot (p_current)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainNorm = p_current_Norm, g = g, glb = glb, gub = gub)
                        #Right foot (p_previous)
                        g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainNorm = p_previous_Norm, g = g, glb = glb, gub = gub)
                        #Friction Cone Constraint
                        #Left foot (p_current)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainTangentX = p_current_TangentX, TerrainTangentY = p_current_TangentY, TerrainNorm = p_current_Norm, miu = miu, g = g, glb = glb, gub = gub)
                        #right foot (p_previous)
                        g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainTangentX = p_previous_TangentX, TerrainTangentY = p_previous_TangentY, TerrainNorm = p_previous_Norm, miu = miu, g = g, glb = glb, gub = gub)

            elif GaitPattern[Nph]== 'Swing':
                #Get contact location
                #In the swing phase, the stance leg is the landing foot of the previous step (Step Number - 1), 
                #but index for the tangents and norm need to +1 as StepNum (in the second level) + 1 = StepNum in the entire horizon
                if StepCnt == 0:#Special Case for the First Step (NOTE:Step 0)
                    p_stance = ca.vertcat(px_init,py_init,pz_init)
                    p_stance_TangentX = SurfTangentsX[0:3]
                    p_stance_TangentY = SurfTangentsY[0:3]
                    p_stance_Norm = SurfNorms[0:3]

                else: #For other Steps, indexed as 1,2,3,4
                    p_stance = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                    p_stance_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                    p_stance_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                    p_stance_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                #Give Constraint according to even and odd steps
                if StepCnt%2 == 0: #Even Number of Steps
                    #------
                    #Case 1
                    #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Left foot is the stance foot

                    #Kinematics Constraint
                    #CoM in the Left (p_stance)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM Height Constraint (Left p_stance foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Center",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL_k,             f_length = F_bound,
                                                            x_p_bar = cLx_p[k], x_q_bar = cLx_q[k], y_p_bar = cLy_p[k], y_q_bar = cLy_q[k], 
                                                            z_p_bar = cLz_p[k], z_q_bar = cLz_q[k], g = g, glb = glb, gub = gub)
                        #Zero Ponton Terms
                        #For Right Foot Contact Points
                        ponton_term_vec = ca.vertcat(cRx_p[k], cRx_q[k], cRy_p[k], cRy_q[k], cRz_p[k], cRz_q[k])
                        g.append(ca.if_else(SwingLegFlag,ponton_term_vec,np.zeros(6)))
                        glb.append(np.zeros(6))
                        gub.append(np.zeros(6))

                    #Zero Forces (Right Foot)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraints on Left Foot p_stance
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint on Left Foot p_stance
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)

                    #--------------
                    #Case 2
                    #If First Level Swing the Right, then the second level Even Number Phases (the first Phase) Swing the Left
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Right foot is the stance foot

                    #Kinematics Constraint
                    #CoM in the Right (p_stance)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                    #CoM Height Constraint (Right p_stance foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Right Foot 
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Center",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR_k,             f_length = F_bound,
                                                            x_p_bar = cRx_p[k], x_q_bar = cRx_q[k], y_p_bar = cRy_p[k], y_q_bar = cRy_q[k], 
                                                            z_p_bar = cRz_p[k], z_q_bar = cRz_q[k], g = g, glb = glb, gub = gub)
                        #Zero Ponton Terms
                        #For Left Foot Contact Points
                        ponton_term_vec = ca.vertcat(cLx_p[k], cLx_q[k], cLy_p[k], cLy_q[k], cLz_p[k], cLz_q[k])
                        g.append(ca.if_else(SwingLegFlag,ponton_term_vec,np.zeros(6)))
                        glb.append(np.zeros(6))
                        gub.append(np.zeros(6))

                    #Zero Forces (Left Foot)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraints on Right Foot p_stance
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint on Right Foot p_stance
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)

                elif StepCnt%2 == 1: #Odd Number of Steps
                    #--------
                    #Case 1
                    #If First Level Swing the Left, then the second level Odd Number Steps Swing the Left
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Right Foot is the stance foot

                    #Kinematics Constraint
                    #CoM in the Right (p_stance)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_stance foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Right Foot
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Center",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR_k,             f_length = F_bound,
                                                            x_p_bar = cRx_p[k], x_q_bar = cRx_q[k], y_p_bar = cRy_p[k], y_q_bar = cRy_q[k], 
                                                            z_p_bar = cRz_p[k], z_q_bar = cRz_q[k], g = g, glb = glb, gub = gub)
                        #Zero Ponton Terms
                        #For Left Foot Contact Points
                        ponton_term_vec = ca.vertcat(cLx_p[k], cLx_q[k], cLy_p[k], cLy_q[k], cLz_p[k], cLz_q[k])
                        g.append(ca.if_else(SwingLegFlag,ponton_term_vec,np.zeros(6)))
                        glb.append(np.zeros(6))
                        gub.append(np.zeros(6))

                    #Zero Forces (Left Foot)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FL_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraints on Right Foot p_stance
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint on Right Foot p_stance
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)

                    #-------
                    #Case 2
                    #If First Level Swing the Right, then the second level Odd Number Steps Swing the Right
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Left foot is the stance foot

                    #Kinematics Constraint
                    #CoM in the Left (p_stance)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Left p_stance foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stance, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Center",   P = p_stance,   P_TangentX = p_stance_TangentX,   P_TangentY = p_stance_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL_k,             f_length = F_bound,
                                                            x_p_bar = cLx_p[k], x_q_bar = cLx_q[k], y_p_bar = cLy_p[k], y_q_bar = cLy_q[k], 
                                                            z_p_bar = cLz_p[k], z_q_bar = cLz_q[k], g = g, glb = glb, gub = gub)
                        #Zero Ponton Terms
                        #For Right Foot Contact Points
                        ponton_term_vec = ca.vertcat(cRx_p[k], cRx_q[k], cRy_p[k], cRy_q[k], cRz_p[k], cRz_q[k])
                        g.append(ca.if_else(SwingLegFlag,ponton_term_vec,np.zeros(6)))
                        glb.append(np.zeros(6))
                        gub.append(np.zeros(6))

                    #Zero Forces (Right Foot)
                    g, glb, gub = ZeroForces(SwingLegIndicator = SwingLegFlag, F_k = FR_k, g = g, glb = glb, gub = gub)
                    #Unilateral Constraints on Left Foot p_stance
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainNorm = p_stance_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint on Left Foot p_stance
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainTangentX = p_stance_TangentX, TerrainTangentY = p_stance_TangentY, TerrainNorm = p_stance_Norm, miu = miu, g = g, glb = glb, gub = gub)

            elif GaitPattern[Nph]=='DoubleSupport':
                #Get contact location
                #In the Double Support Phase, the p_stationary is the foot is the un-moved foot during StepNum (in second level)
                #the p_land is the landing/moving foot during StepNum (in the second level)
                if StepCnt == 0: #Special Case for the First Step (NOTE: Step 0)
                    p_stationary = ca.vertcat(px_init,py_init,pz_init)
                    p_stationary_TangentX = SurfTangentsX[0:3]
                    p_stationary_TangentY = SurfTangentsY[0:3]
                    p_stationary_Norm = SurfNorms[0:3]

                    p_land = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                    p_land_TangentX = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_TangentY = SurfTangentsY[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_Norm = SurfNorms[(StepCnt+1)*3:(StepCnt+1)*3+3]
            
                else: #For other steps, indexed as 1,2,3,4
                    #In the Double Support Phase, the p_stationary is the foot is the un-moved foot during StepNum (in second level), \
                    #which is StepCnt - 1 (the moving foot in the previous step)
                    #the p_land is the landing/moving foot during StepNum (in the second level) --- StepCnt
                    #The StepCnt + 1 is the StepNum in the entire horizon (for getting terrain tangents and norm)
                    p_stationary = ca.vertcat(px[StepCnt-1],py[StepCnt-1],pz[StepCnt-1])
                    p_stationary_TangentX = SurfTangentsX[StepCnt*3:StepCnt*3+3]
                    p_stationary_TangentY = SurfTangentsY[StepCnt*3:StepCnt*3+3]
                    p_stationary_Norm = SurfNorms[StepCnt*3:StepCnt*3+3]

                    p_land = ca.vertcat(px[StepCnt],py[StepCnt],pz[StepCnt])
                    p_land_TangentX = SurfTangentsX[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_TangentY = SurfTangentsY[(StepCnt+1)*3:(StepCnt+1)*3+3]
                    p_land_Norm = SurfNorms[(StepCnt+1)*3:(StepCnt+1)*3+3]

                #Give Constraint according to even and odd steps
                if StepCnt%2 == 0: #Even Number of Steps
                    #-----------
                    #Case 1
                    #If First Level Swing the Left, then the second level Even Steps Swing the Right
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Left Foot is stationary
                    #Right Foot is the Land

                    #Kinemactics Constraint
                    #CoM in the Left (p_stationary)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM in the Right (p_land)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                    
                    #   CoM Height Constraint (Left p_stationary foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_land foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Center",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL_k,             f_length = F_bound,
                                                            x_p_bar = cLx_p[k], x_q_bar = cLx_q[k], y_p_bar = cLy_p[k], y_q_bar = cLy_q[k], 
                                                            z_p_bar = cLz_p[k], z_q_bar = cLz_q[k], g = g, glb = glb, gub = gub)
                        #--------------------------
                        #Right Foot
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Center",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR_k,             f_length = F_bound,
                                                            x_p_bar = cRx_p[k], x_q_bar = cRx_q[k], y_p_bar = cRy_p[k], y_q_bar = cRy_q[k], 
                                                            z_p_bar = cRz_p[k], z_q_bar = cRz_q[k], g = g, glb = glb, gub = gub)
                    #Unilateral Constraint
                    #Left foot obey the unilateral constraint on p_stationary
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the Stationary foot p_land
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #Left foot obey the friction cone constraint on p_stationary
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the on p_land
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)

                    #------------
                    #Case 2
                    #If First Level Swing the Right, then the second level Even Steps Swing the Left
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Right Foot is stationary
                    #Left Foot is the Land

                    #Kinemactics Constraint
                    #CoM in the Left (p_land)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM in the Right (p_stationary)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                    
                    #   CoM Height Constraint (Left p_land foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_stationary foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Center",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL_k,             f_length = F_bound,
                                                            x_p_bar = cLx_p[k], x_q_bar = cLx_q[k], y_p_bar = cLy_p[k], y_q_bar = cLy_q[k], 
                                                            z_p_bar = cLz_p[k], z_q_bar = cLz_q[k], g = g, glb = glb, gub = gub)
                        #--------------------------
                        #Right Foot Contact Point 1
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Center",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR_k,             f_length = F_bound,
                                                            x_p_bar = cRx_p[k], x_q_bar = cRx_q[k], y_p_bar = cRy_p[k], y_q_bar = cRy_q[k], 
                                                            z_p_bar = cRz_p[k], z_q_bar = cRz_q[k], g = g, glb = glb, gub = gub)
                    #Unilateral Constraint
                    #Left foot obey the unilateral constraint on p_land
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the Stationary foot p_stationary
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #Left foot obey the friction cone constraint on p_land
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the on p_stationary
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)

                elif StepCnt%2 == 1:#Odd Number of Steps
                    #------
                    #Case 1
                    #If First Level Swing the Left, then the second level Odd Steps Swing the Left
                    SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
                    #Right Foot is the Stationary
                    #Left Foot is the Land

                    #Kinemactics Constraint
                    #CoM in the Left (p_land)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM in the Right (p_stationary)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Left p_land foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_stationary foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Center",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL_k,             f_length = F_bound,
                                                            x_p_bar = cLx_p[k], x_q_bar = cLx_q[k], y_p_bar = cLy_p[k], y_q_bar = cLy_q[k], 
                                                            z_p_bar = cLz_p[k], z_q_bar = cLz_q[k], g = g, glb = glb, gub = gub)
                        #--------------------------
                        #Right Foot
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Center",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR_k,             f_length = F_bound,
                                                            x_p_bar = cRx_p[k], x_q_bar = cRx_q[k], y_p_bar = cRy_p[k], y_q_bar = cRy_q[k], 
                                                            z_p_bar = cRz_p[k], z_q_bar = cRz_q[k], g = g, glb = glb, gub = gub)
                    #Unilateral Constraint
                    #Left foot obey the unilateral constraint on p_land
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the Stationary foot p_stationary
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #Left foot obey the friction cone constraint on p_land
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the on p_stationary
                    g, glb, gub = FrictionCone(SwingLegIndicator = ParaLeftSwingFlag, F_k = FR_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
  
                    #------
                    #Case 2
                    #If First Level Swing the Right, then the second level Odd Steps Swing the Right
                    SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
                    #Left Foot is the stationary
                    #Right Foot is the Land

                    #Kinematics Constraint
                    #CoM in the Left (p_stationary)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, K_polytope = K_CoM_Left, k_polytope = k_CoM_Left, g = g, glb = glb, gub = gub)
                    #CoM in the Right (p_land)
                    g, glb, gub = CoM_Kinematics(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, K_polytope = K_CoM_Right, k_polytope = k_CoM_Right, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Left p_stationary foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_stationary, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)
                    #   CoM Height Constraint (Right p_land foot)
                    g, glb, gub = CoM_to_Foot_Height_Limit(SwingLegIndicator = SwingLegFlag, CoM_k = CoM_k, P = p_land, h_min = CoM_z_to_Foot_min, h_max = CoM_z_to_Foot_max, g = g, glb = glb, gub = gub)

                    #Angular Dynamics (Ponton)
                    if k<N_K-1: #double check the knot number is valid
                        #------------------------------
                        #Left Foot
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Center",   P = p_stationary,   P_TangentX = p_stationary_TangentX,   P_TangentY = p_stationary_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FL_k,             f_length = F_bound,
                                                            x_p_bar = cLx_p[k], x_q_bar = cLx_q[k], y_p_bar = cLy_p[k], y_q_bar = cLy_q[k], 
                                                            z_p_bar = cLz_p[k], z_q_bar = cLz_q[k], g = g, glb = glb, gub = gub)
                        #--------------------------
                        #Right Foot
                        g, glb, gub = Ponton_Concex_Constraint(SwingLegIndicator = SwingLegFlag,
                                                            P_name = "Center",   P = p_land,   P_TangentX = p_land_TangentX,   P_TangentY = p_land_TangentY,
                                                            CoM_k = CoM_k,         l_length = max_leg_length,
                                                            f = FR_k,             f_length = F_bound,
                                                            x_p_bar = cRx_p[k], x_q_bar = cRx_q[k], y_p_bar = cRy_p[k], y_q_bar = cRy_q[k], 
                                                            z_p_bar = cRz_p[k], z_q_bar = cRz_q[k], g = g, glb = glb, gub = gub)
                    #Unilateral Constraint
                    #Left foot obey the unilateral constraint on p_stationary
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainNorm = p_stationary_Norm, g = g, glb = glb, gub = gub)
                    #then the Right foot is obey the unilateral constraint on the Stationary foot p_land
                    g, glb, gub = Unilateral_Constraints(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainNorm = p_land_Norm, g = g, glb = glb, gub = gub)
                    #Friction Cone Constraint
                    #Left foot obey the friction cone constraint on p_stationary
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FL_k, TerrainTangentX = p_stationary_TangentX, TerrainTangentY = p_stationary_TangentY, TerrainNorm = p_stationary_Norm, miu = miu, g = g, glb = glb, gub = gub)
                    #then the right foot obeys the friction cone constraints on the on p_land
                    g, glb, gub = FrictionCone(SwingLegIndicator = SwingLegFlag, F_k = FR_k, TerrainTangentX = p_land_TangentX, TerrainTangentY = p_land_TangentY, TerrainNorm = p_land_Norm, miu = miu, g = g, glb = glb, gub = gub)

                    #-------------------------------------
            else:
                raise Exception("Unknown Phase Name")                   

            #-------------------------------------
            #Dynamics Constraint
            if k < N_K - 1: #N_K - 1 the enumeration of the last knot, -1 the knot before the last knot
                #First-order Dynamics CoM x, y, z
                g, glb, gub = First_Order_Integrator(next_state = x[k+1], cur_state = x[k], cur_derivative = xdot[k], h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = y[k+1], cur_state = y[k], cur_derivative = ydot[k], h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = z[k+1], cur_state = z[k], cur_derivative = zdot[k], h = h, g = g, glb = glb, gub = gub)
                #First-order Dynamics CoMdot x, y, z
                g, glb, gub = First_Order_Integrator(next_state = xdot[k+1], cur_state = xdot[k], 
                                                     cur_derivative = FLx[k]/m + FRx[k]/m, 
                                                     h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = ydot[k+1], cur_state = ydot[k], 
                                                     cur_derivative = FLy[k]/m + FRy[k]/m, 
                                                     h = h, g = g, glb = glb, gub = gub)
                g, glb, gub = First_Order_Integrator(next_state = zdot[k+1], cur_state = zdot[k], 
                                                     cur_derivative = FLz[k]/m + FRz[k]/m - G, 
                                                     h = h, g = g, glb = glb, gub = gub)

            #Cost Terms
            if k < N_K - 1:
                #with Acceleration
                J = J + h*(FLx[k]/m + FRx[k]/m)**2 + \
                        h*(FLy[k]/m + FRy[k]/m)**2 + \
                        h*(FLz[k]/m + FRz[k]/m - G)**2
                #with Angular Momentum Rate (Ponton)
                J = J + h*(cLx_p[k]**2 + cLx_q[k]**2 + cLy_p[k]**2 + cLy_q[k]**2 + cLz_p[k]**2 + cLz_q[k]**2 + \
                           cRx_p[k]**2 + cRx_q[k]**2 + cRy_p[k]**2 + cRy_q[k]**2 + cRz_p[k]**2 + cRz_q[k]**2)
    
    #-------------------------------------
    #Relative Footstep Constraint
    for step_cnt in range(Nsteps):
        if step_cnt == 0:
            #!!!!!!Pass from the first Level!!!!!!
            P_k_current = ca.vertcat(px_init,py_init,pz_init) #ca.vertcat(px[-1],py[-1],pz[-1])
            #!!!!!!
            P_k_next = ca.vertcat(px[step_cnt],py[step_cnt],pz[step_cnt])
        else:
            P_k_current = ca.vertcat(px[step_cnt-1],py[step_cnt-1],pz[step_cnt-1])
            P_k_next = ca.vertcat(px[step_cnt],py[step_cnt],pz[step_cnt])

        if step_cnt%2 == 0: #even number steps
            #----
            #Case 1
            #If First Level Swing the Left, then the second level Even Number Phases (the first Phase) Swing the Right
            SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
            #Right Foot is landing (p_next), Left foot is stationary (p_current)
            #Relative Swing Foot Location - Landing (p_next/Right) in Stationary (p_current/Left) (rf in lf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_next, p_cur = P_k_current, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, 
                                     g = g, glb = glb, gub = gub)
            #Relative Swing Foot Location - Stationary (p_current/Left) in Landing (p_next/Right) (lf in rf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_current, p_cur = P_k_next, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, 
                                     g = g, glb = glb, gub = gub)

            #------
            #Case 2
            #If First Levvel Swing the Right, then the second level Even Number Phases (the first phase) Swing the Left
            SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
            #Left foot is landing (p_next), Right Foot is stationary (p_current)
            #Relative Swing Foot Location - Landing (p_next/Left) in stationary (p_current/Right) (lf in rf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_next, p_cur = P_k_current, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, 
                                     g = g, glb = glb, gub = gub)
            #Relative Swing Foot Location - Stationary (p_current/Right) in Landing (p_next/Left) (rf in lf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_current, p_cur = P_k_next, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, 
                                     g = g, glb = glb, gub = gub)

        elif step_cnt%2 == 1: #odd number steps
            #-------
            #Case 1
            #If the first level swings the Left, then the second level for Odd Number Phases (the second phase) swings left
            SwingLegFlag = ParaLeftSwingFlag #Indicator of which leg is swinging
            #Left Foot is landing (p_next), Right Foot is stationary (p_current)
            #Relative Swing Foot Location - Landing (p_next/Left) in stationary (p_current/Right) (lf in rf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_next, p_cur = P_k_current, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, 
                                     g = g, glb = glb, gub = gub)
            #Relative Swing Foot Location - Stationary (p_current/Right) in Landing (p_next/Left) (rf in lf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_current, p_cur = P_k_next, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, 
                                     g = g, glb = glb, gub = gub)

            #-------
            #Case 2
            #If the first level swings the Right, then the second level for Odd Number Phases (the second phase) swings Right
            SwingLegFlag = ParaRightSwingFlag #Indicator of which leg is swinging
            #Right Foot is landing (p_next), Left foot is stationary (p_current)
            #Relative Swing Foot Location - Landing (p_next/Right) in Stationary (p_current/Left) (rf in lf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_next, p_cur = P_k_current, Q_polytope = Q_rf_in_lf, q_polytope = q_rf_in_lf, 
                                     g = g, glb = glb, gub = gub)
            #Relative Swing Foot Location - Stationary (p_current/Left) in Landing (p_next/Right) (lf in rf)
            Relative_Foot_Kinematics(SwingLegIndicator = SwingLegFlag, p_next = P_k_current, p_cur = P_k_next, Q_polytope = Q_lf_in_rf, q_polytope = q_lf_in_rf, 
                                     g = g, glb = glb, gub = gub)

    #-----------------
    #FootStep Location Constraint (On the Patch) -> Only One Step
    #P3----------------P1
    #|                  |
    #|                  |
    #|                  |
    #P4----------------P2

    for PatchNum in range(Nsteps): #No need to consider p_init, as they are constrained by the first level
        #Get Footstep Vector
        P_vector = ca.vertcat(px[PatchNum],py[PatchNum],pz[PatchNum])

        #Get Half Space Representation 
        #NOTE: In the second level, the terrain patch start from the second patch, indexed as 1
        SurfParaTemp = SurfParas[20+PatchNum*20:19+(PatchNum+1)*20+1]
        #print(SurfParaTemp)
        SurfK = SurfParaTemp[0:11+1]
        SurfK = ca.reshape(SurfK,3,4)
        SurfK = SurfK.T #NOTE: This process is due to casadi naming convention to have first row to be x1,x2,x3
        SurfE = SurfParaTemp[11+1:11+3+1]
        Surfk = SurfParaTemp[14+1:14+4+1]
        Surfe = SurfParaTemp[-1]

        #Terrain Tangent and Norms
        #In the second level, when getting the tangets, we need to have PatchNum/StepCnt + 1, 
        #As the second level counts the second step (enumerated as step 1 in entire horizon) as step 0
        P_vector_TangentX = SurfTangentsX[(PatchNum+1)*3:(PatchNum+1)*3+3]
        P_vector_TangentY = SurfTangentsY[(PatchNum+1)*3:(PatchNum+1)*3+3]

        g, glb, gub = Stay_on_Surf(P = P_vector, P_TangentX = P_vector_TangentX, P_TangentY = P_vector_TangentY, 
                                   ineq_K = SurfK, ineq_k = Surfk, eq_E = SurfE, eq_e = Surfe, g = g, glb = glb, gub = gub)

    # #-----------------------------------
    # #Switching Time Constraint
    # for phase_cnt in range(Nphase):
    #     if GaitPattern[phase_cnt] == 'InitialDouble':
    #         if phase_cnt == 0:
    #             g.append(Ts[phase_cnt] - 0)
    #             glb.append(np.array([0.1])) #old 0.1-0.3
    #             gub.append(np.array([0.3]))
    #         else:
    #             g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
    #             glb.append(np.array([0.1]))
    #             gub.append(np.array([0.3]))
    #     elif GaitPattern[phase_cnt] == 'Swing':
    #         g.append(Ts[phase_cnt]-Ts[phase_cnt-1])
    #         glb.append(np.array([0.8])) #0.8-1.2
    #         gub.append(np.array([1.2]))
    #     elif GaitPattern[phase_cnt] == 'DoubleSupport':
    #         g.append(Ts[phase_cnt]-Ts[phase_cnt-1]) #0.1-0.9
    #         glb.append(np.array([0.1]))
    #         gub.append(np.array([0.3])) #old - 0.1-0.3
    #     else:
    #         raise Exception("Unknown Phase Name")

    #-----------------------------------------------------------------------------------------------------------------------
    #Get Variable Index - !!!This is the pure Index, when try to get the array using other routines, we need to add "+1" at the last index due to Python indexing conventions
    #First set of variables start counting from 0, The enumeration of the last knot is N_K-1
    x_index = (0,N_K-1);                                      y_index = (x_index[1]+1,x_index[1]+N_K);                 z_index = (y_index[1]+1,y_index[1]+N_K)
    xdot_index = (z_index[1]+1,z_index[1]+N_K);               ydot_index = (xdot_index[1]+1,xdot_index[1]+N_K);        zdot_index = (ydot_index[1]+1,ydot_index[1]+N_K)
    FLx_index = (zdot_index[1]+1,zdot_index[1]+N_K);          FLy_index = (FLx_index[1]+1,FLx_index[1]+N_K);           FLz_index = (FLy_index[1]+1,FLy_index[1]+N_K)
    FRx_index = (FLz_index[1]+1,FLz_index[1]+N_K);            FRy_index = (FRx_index[1]+1,FRx_index[1]+N_K);           FRz_index = (FRy_index[1]+1,FRy_index[1]+N_K)
    px_init_index = (FRz_index[1]+1,FRz_index[1]+1);          py_init_index = (px_init_index[1]+1,px_init_index[1]+1); pz_init_index = (py_init_index[1]+1,py_init_index[1]+1)
    px_index = (pz_init_index[1]+1,pz_init_index[1]+Nsteps);  py_index = (px_index[1]+1,px_index[1]+Nsteps);           pz_index = (py_index[1]+1,py_index[1]+Nsteps)
    Ts_index = (pz_index[1]+1,pz_index[1]+Nphase)
    
    cLx_p_index = (Ts_index[1]+1,Ts_index[1]+N_K);            cLx_q_index = (cLx_p_index[1]+1,cLx_p_index[1]+N_K)
    cLy_p_index = (cLx_q_index[1]+1,cLx_q_index[1]+N_K);      cLy_q_index = (cLy_p_index[1]+1,cLy_p_index[1]+N_K)
    cLz_p_index = (cLy_q_index[1]+1,cLy_q_index[1]+N_K);      cLz_q_index = (cLz_p_index[1]+1,cLz_p_index[1]+N_K)
    
    cRx_p_index = (cLz_q_index[1]+1,cLz_q_index[1]+N_K);      cRx_q_index = (cRx_p_index[1]+1,cRx_p_index[1]+N_K)
    cRy_p_index = (cRx_q_index[1]+1,cRx_q_index[1]+N_K);      cRy_q_index = (cRy_p_index[1]+1,cRy_p_index[1]+N_K)
    cRz_p_index = (cRy_q_index[1]+1,cRy_q_index[1]+N_K);      cRz_q_index = (cRz_p_index[1]+1,cRz_p_index[1]+N_K)

    var_index = {"x":x_index,        "y":y_index,         "z":z_index,         "xdot":xdot_index,     "ydot":ydot_index,     "zdot":zdot_index,
                 "FLx":FLx_index,    "FLy":FLy_index,     "FLz":FLz_index,   
                 "FRx":FRx_index,    "FRy":FRy_index,     "FRz":FRz_index,
                 "px_init":px_init_index,   "py_init":py_init_index,   "pz_init":pz_init_index,
                 "px":px_index,   "py":py_index,   "pz":pz_index,
                 "Ts":Ts_index,
                 "cLx_p":cLx_p_index,    "cLx_q":cLx_q_index,    "cLy_p":cLy_p_index,    "cLy_q":cLy_q_index,    "cLz_p":cLz_p_index,    "cLz_q":cLz_q_index,
                 "cRx_p":cRx_p_index,    "cRx_q":cRx_q_index,    "cRy_p":cRy_p_index,    "cRy_q":cRy_q_index,    "cRz_p":cRz_p_index,    "cRz_q":cRz_q_index,  
    }

    return DecisionVars, DecisionVars_lb, DecisionVars_ub, J, g, glb, gub, var_index

#Build OCP Solver
#   NumSurfaces is the same as the total number of steps(TotalNumSteps)
#   ObjReachingType defines how do we want to reach local obj state (tracking local obj should be only allowed for single step NLP)
#      "cost":         tracking the local objective via cost 
#      "constraints":   reach a pre-defined state (CoM + FootStep Location)
#      None:           means we reach the far goal, which is will fail for sure
def ocp_solver_build(FirstLevel = None, SecondLevel = None, TotalNumSteps = None, LocalObjTrackingType = None, N_knots_local = 7, robot_mass = 100.0, PhaseDurationLimits = None, miu = 0.3, max_compute_time = 1000.0):
    #Check if the First Level is selected properly
    assert FirstLevel != None, "First Level is Not Selected."
    assert TotalNumSteps != None, "Total Number of Steps Un-defined."

    #Define Default Phase Duration Limits
    if PhaseDurationLimits == None:
        PhaseDurationLimits = {"DoubleSupport_Min": 0.05, "DoubleSupport_Max": 0.3,
                               "SingleSupport_Min": 0.8,  "SingleSupport_Max": 1.2}

    #Double confirm LocalObjTrackingType is None when we have multiplesteps NLP
    if TotalNumSteps > 1:
        LocalObjTrackingType = None

    #Number of Surfaces, equivalent to total number of steps
    NumSurfaces = TotalNumSteps

    #Define Solver Parameter Vector
    #  Foot Swing Indicators
    ParaLeftSwingFlag = ca.SX.sym('LeftSwingFlag')
    ParaRightSwingFlag = ca.SX.sym('RightSwingFlag')
    #Initial CoM Position
    x_init = ca.SX.sym('x_init');         y_init = ca.SX.sym('y_init');          z_init = ca.SX.sym('z_init')
    #Initial CoM Velocity
    xdot_init = ca.SX.sym('xdot_init');   ydot_init = ca.SX.sym('ydot_init');    zdot_init = ca.SX.sym('zdot_init')
    #Initial Angular Momentum
    Lx_init = ca.SX.sym('Lx_init');       Ly_init = ca.SX.sym('Ly_init');        Lz_init = ca.SX.sym('Lz_init')
    #Initial Angular Momentum Rate (not useful, just put there to be align with TSID script)
    Ldotx_init = ca.SX.sym('Ldotx_init'); Ldoty_init = ca.SX.sym('Ldoty_init');  Ldotz_init = ca.SX.sym('Ldotz_init')
    #Initial Left Foot Position
    PLx_init = ca.SX.sym('PLx_init');     PLy_init = ca.SX.sym('PLy_init');      PLz_init = ca.SX.sym('PLz_init')
    #Initial Right Foot Position
    PRx_init = ca.SX.sym('PRx_init');     PRy_init = ca.SX.sym('PRy_init');      PRz_init = ca.SX.sym('PRz_init')
    #(Far) Goal CoM Position
    x_end = ca.SX.sym('x_end');           y_end = ca.SX.sym('y_end');            z_end = ca.SX.sym('z_end')
    #(Far) Goal CoM Velocity (not useful, just put there to be align with TSID script)
    xdot_end = ca.SX.sym('xdot_end');     ydot_end = ca.SX.sym('ydot_end');      zdot_end = ca.SX.sym('zdot_end')
    #   Surface Parameters 
    SurfParas = []
    for surfNum in range(NumSurfaces):
        SurfTemp = ca.SX.sym('S'+str(surfNum),3*4+3+5);   SurfParas.append(SurfTemp)
        #3-by-4 matrix for ineq, 3 x 1 for eq, 5 numbers for ineq + eq
    SurfParas = ca.vertcat(*SurfParas)

    #   Surface Orientations
    SurfOrientations = []
    for surfNum in range(NumSurfaces):
        SurfOrienTemp = ca.SX.sym('Sorien'+str(surfNum),3*3);   SurfOrientations.append(SurfOrienTemp)
    SurfOrientations = ca.vertcat(*SurfOrientations)

    #   Surface TangentsX, Tangent Y, Norm
    SurfTangentsX = [];   SurfTangentsY = [];   SurfNorms = []
    for surfNum in range(NumSurfaces):
        SurfTangentX_temp = ca.SX.sym('SurfTengentX'+str(surfNum),3);   SurfTangentsX.append(SurfTangentX_temp)
        SurfTangentY_temp = ca.SX.sym('SurfTengentY'+str(surfNum),3);   SurfTangentsY.append(SurfTangentY_temp)
        SurfNorm_temp     = ca.SX.sym('SurfNorm'+str(surfNum),3);       SurfNorms.append(SurfNorm_temp)
    SurfTangentsX = ca.vertcat(*SurfTangentsX);   SurfTangentsY = ca.vertcat(*SurfTangentsY);   SurfNorms = ca.vertcat(*SurfNorms)

    #   Tangents and Norm for initial contacts
    PL_init_TangentX = ca.SX.sym('PL_init_TangentX',3);   PL_init_TangentY = ca.SX.sym('PL_init_TangentY',3);   PL_init_Norm = ca.SX.sym('PL_init_Norm',3)
    PR_init_TangentX = ca.SX.sym('PR_init_TangentX',3);   PR_init_TangentY = ca.SX.sym('PR_init_TangentY',3);   PR_init_Norm = ca.SX.sym('PR_init_Norm',3)
    
    #   Initial Contact Patch Orientations
    PL_init_Orientation = ca.SX.sym('PL_init_Orientation',3*3);     PR_init_Orientation = ca.SX.sym('PR_init_Orientation',3*3)

    #   Local Obj
    #   Local Obj CoM State
    x_obj = ca.SX.sym("x_obj");          y_obj = ca.SX.sym("y_obj");         z_obj = ca.SX.sym("z_obj")
    #   Local Obj CoM Velocity
    xdot_obj = ca.SX.sym("xdot_obj");    ydot_obj = ca.SX.sym("ydot_obj");   zdot_obj = ca.SX.sym("zdot_obj")
    #   Local Obj Angular Momentum
    Lx_obj = ca.SX.sym("Lx_obj");        Ly_obj = ca.SX.sym("Ly_obj");       Lz_obj = ca.SX.sym("Lz_obj")
    #   Target Contact Location
    Px_obj = ca.SX.sym("Px_obj");        Py_obj = ca.SX.sym("Py_obj");       Pz_obj = ca.SX.sym("Pz_obj")
    #   Target Phase Duration
    InitDS_Ts_obj = ca.SX.sym("InitDS_Ts_obj") #Init Double Support
    SS_Ts_obj = ca.SX.sym("SS_Ts_obj") #Single Support
    DS_Ts_obj = ca.SX.sym("DS_Ts_obj") #Double Support

    #Collect Casadi Parameters
    #Make a dictionary to for usage in ocp constructions (i.e. nlp-single step or nlp-second-level)
    ParaList = {"LeftSwingFlag":ParaLeftSwingFlag,   "RightSwingFlag":ParaRightSwingFlag,
                "x_init":x_init,          "y_init":y_init,                 "z_init":z_init,
                "xdot_init":xdot_init,    "ydot_init":ydot_init,           "zdot_init":zdot_init,
                "Lx_init":Lx_init,        "Ly_init":Ly_init,               "Lz_init":Lz_init,
                "Ldotx_init":Ldotx_init,  "Ldoty_init":Ldoty_init,         "Ldotz_init":Ldotz_init,
                "PLx_init":PLx_init,      "PLy_init":PLy_init,             "PLz_init":PLz_init,
                "PRx_init":PRx_init,      "PRy_init":PRy_init,             "PRz_init":PRz_init,
                "x_end":x_end,            "y_end":y_end,                   "z_end":z_end,
                "xdot_end":xdot_end,      "ydot_end":ydot_end,             "zdot_end":zdot_end,
                "SurfParas":SurfParas,    "SurfTangentsX":SurfTangentsX,   "SurfTangentsY":SurfTangentsY,    "SurfNorms":SurfNorms,   "SurfOrientations":SurfOrientations,
                "PL_init_TangentX":PL_init_TangentX,  "PL_init_TangentY":PL_init_TangentY,  "PL_init_Norm":PL_init_Norm,  "PL_init_Orientation": PL_init_Orientation,
                "PR_init_TangentX":PR_init_TangentX,  "PR_init_TangentY":PR_init_TangentY,  "PR_init_Norm":PR_init_Norm,  "PR_init_Orientation": PR_init_Orientation,
                "x_obj": x_obj,           "y_obj":y_obj,                   "z_obj":z_obj,
                "xdot_obj":xdot_obj,      "ydot_obj":ydot_obj,             "zdot_obj":zdot_obj,
                "Lx_obj":Lx_obj,          "Ly_obj":Ly_obj,                 "Lz_obj":Lz_obj,
                "Px_obj":Px_obj,          "Py_obj":Py_obj,                 "Pz_obj":Pz_obj,
                "InitDS_Ts_obj":InitDS_Ts_obj, "SS_Ts_obj":SS_Ts_obj, "DS_Ts_obj":DS_Ts_obj, 
    }
    
    paras = ca.vertcat(ParaLeftSwingFlag, ParaRightSwingFlag,
                       x_init,        y_init,        z_init,
                       xdot_init,     ydot_init,     zdot_init,
                       Lx_init,       Ly_init,       Lz_init,
                       Ldotx_init,    Ldoty_init,    Ldotz_init,
                       PLx_init,      PLy_init,      PLz_init,
                       PRx_init,      PRy_init,      PRz_init,
                       x_end,         y_end,         z_end,
                       xdot_end,      ydot_end,      zdot_end,
                       SurfParas,     SurfTangentsX, SurfTangentsY,  SurfNorms,    SurfOrientations,
                       PL_init_TangentX,   PL_init_TangentY,   PL_init_Norm,   PL_init_Orientation,
                       PR_init_TangentX,   PR_init_TangentY,   PR_init_Norm,   PR_init_Orientation,
                       x_obj,         y_obj,         z_obj,
                       xdot_obj,      ydot_obj,      zdot_obj,
                       Lx_obj,        Ly_obj,        Lz_obj,
                       Px_obj,        Py_obj,        Pz_obj,
                       InitDS_Ts_obj, SS_Ts_obj,     DS_Ts_obj)

    #print(paras.shape)

    #Build the solver, give connecting constraints
    #-------------
    #Make the first step NLP
    if FirstLevel == "NLP_SingleStep":
        if TotalNumSteps == 1: #Local obj mode
            #make the first level
            var_lv1, var_lb_lv1, var_ub_lv1, J_lv1, g_lv1, glb_lv1, gub_lv1, var_idx_lv1 = NLP_SingleStep(ParameterList = ParaList, Nk_Local = N_knots_local, m = robot_mass, PhaseDuration_Limits = PhaseDurationLimits, miu = miu, LocalObjMode=True)
        else:
            var_lv1, var_lb_lv1, var_ub_lv1, J_lv1, g_lv1, glb_lv1, gub_lv1, var_idx_lv1 = NLP_SingleStep(ParameterList = ParaList, Nk_Local = N_knots_local, m = robot_mass, PhaseDuration_Limits = PhaseDurationLimits, miu = miu, LocalObjMode=False)
    else:
        raise Exception("Unknown First Level Name")
    #----------
    #Make the Second Level
    if TotalNumSteps == 1:#No Second Level, So all containers are empty
         var_lv2 = [];   var_lb_lv2 = [];   var_ub_lv2 = []
         J_lv2 = 0.0
         g_lv2 = [];     glb_lv2 = [];      gub_lv2 = []
         var_idx_lv2 = []
    elif TotalNumSteps > 1:
        if SecondLevel == "NLP_SecondLevel": #Number of Step = Total Number of Step - 1 as the first step is included in the first level
            var_lv2, var_lb_lv2, var_ub_lv2, J_lv2, g_lv2, glb_lv2, gub_lv2, var_idx_lv2 = NLP_SecondLevel(ParameterList = ParaList, Nsteps = TotalNumSteps-1, Nk_Local = N_knots_local, m = robot_mass, PhaseDuration_Limits = PhaseDurationLimits, miu = miu)
        elif SecondLevel == "Ponton_FourPoints":
            var_lv2, var_lb_lv2, var_ub_lv2, J_lv2, g_lv2, glb_lv2, gub_lv2, var_idx_lv2 = Ponton_FourPoints(ParameterList = ParaList, Nsteps = TotalNumSteps-1, Nk_Local = N_knots_local, m = robot_mass, PhaseDuration_Limits = PhaseDurationLimits)
        elif SecondLevel == "Ponton_SinglePoint":
            var_lv2, var_lb_lv2, var_ub_lv2, J_lv2, g_lv2, glb_lv2, gub_lv2, var_idx_lv2 = Ponton_SinglePoint(ParameterList = ParaList, Nsteps = TotalNumSteps-1, Nk_Local = N_knots_local, m = robot_mass, PhaseDuration_Limits = PhaseDurationLimits)
        else:
            raise Exception("Unknown Second Level Name")
    #-----------
    #Collect All Decision Vars, Lower and Upper Var Bounds
    #   Decision Vars, lower and upper bounds
    DecisionVars = ca.vertcat(var_lv1, var_lv2)
    DecisionVars_lb = np.concatenate((var_lb_lv1,var_lb_lv2),axis=None)
    DecisionVars_ub = np.concatenate((var_ub_lv1,var_ub_lv2),axis=None)

    #-------------
    #Collect All Decision Vars Index
    var_index = {"Level1_Var_Index": var_idx_lv1,
                 "Level2_Var_Index": var_idx_lv2}

    #------------------
    #Build Cost
    J = 0.0
    #   Construct Terminal Cost and Local Obj Cost
    J_terminal = 0.0;    J_localobj = 0.0
    #   Get Terminal States
    if TotalNumSteps == 1: #Single Step NLP, terminal state all in the first level
        #Get Terminal CoM state of the first Level
        x_lv1 = var_lv1[var_idx_lv1["x"][0]:var_idx_lv1["x"][1]+1];   x_T = x_lv1[-1]
        y_lv1 = var_lv1[var_idx_lv1["y"][0]:var_idx_lv1["y"][1]+1];   y_T = y_lv1[-1]
        z_lv1 = var_lv1[var_idx_lv1["z"][0]:var_idx_lv1["z"][1]+1];   z_T = z_lv1[-1]
        #Get Terminal CoMdot State of the first level
        xdot_lv1 = var_lv1[var_idx_lv1["xdot"][0]:var_idx_lv1["xdot"][1]+1];   xdot_T = xdot_lv1[-1]
        ydot_lv1 = var_lv1[var_idx_lv1["ydot"][0]:var_idx_lv1["ydot"][1]+1];   ydot_T = ydot_lv1[-1]
        zdot_lv1 = var_lv1[var_idx_lv1["zdot"][0]:var_idx_lv1["zdot"][1]+1];   zdot_T = zdot_lv1[-1]
        #Get Terminal Angular Momentum of the First level
        Lx_lv1 = var_lv1[var_idx_lv1["Lx"][0]:var_idx_lv1["Lx"][1]+1];   Lx_T = Lx_lv1[-1]
        Ly_lv1 = var_lv1[var_idx_lv1["Ly"][0]:var_idx_lv1["Ly"][1]+1];   Ly_T = Ly_lv1[-1]
        Lz_lv1 = var_lv1[var_idx_lv1["Lz"][0]:var_idx_lv1["Lz"][1]+1];   Lz_T = Lz_lv1[-1]
        #Get the Contact Location of the first Level
        Px_lv1 = var_lv1[var_idx_lv1["px"][-1]];   Py_lv1 = var_lv1[var_idx_lv1["py"][-1]];     Pz_lv1 = var_lv1[var_idx_lv1["pz"][-1]]
        #Get the Timing vector
        Ts_lv1_vector = var_lv1[var_idx_lv1["Ts"][0]:var_idx_lv1["Ts"][1]+1]
    elif TotalNumSteps > 1: #Multi Step NLP, terminal state all in the second level
        #Get the CoM Terminal State from the second level
        x_lv2 = var_lv2[var_idx_lv2["x"][0]:var_idx_lv2["x"][1]+1];   x_T = x_lv2[-1]
        y_lv2 = var_lv2[var_idx_lv2["y"][0]:var_idx_lv2["y"][1]+1];   y_T = y_lv2[-1]
        z_lv2 = var_lv2[var_idx_lv2["z"][0]:var_idx_lv2["z"][1]+1];   z_T = z_lv2[-1]

    #Build Terminal or Local obj Cost
    if (TotalNumSteps == 1) and (LocalObjTrackingType == "cost"):#Single Step NLP with Local Obj Tracking
        weight = 10000.0
        # #For un-fixed Timing
        # J_localobj = weight*(x_T    - x_obj)**2    + weight*(y_T    - y_obj)**2    + weight*(z_T    - z_obj)**2    + \
        #              weight*(xdot_T - xdot_obj)**2 + weight*(ydot_T - ydot_obj)**2 + weight*(zdot_T - zdot_obj)**2 + \
        #              weight*(Lx_T   - Lx_obj)**2   + weight*(Ly_T   - Ly_obj)**2   + weight*(Lz_T   - Lz_obj)**2 + \
        #              weight*(Px_lv1 - Px_obj)**2   + weight*(Py_lv1 - Py_obj)**2   + weight*(Pz_lv1 - Pz_obj)**2 + \
        #              weight*(Ts_lv1_vector[0] - InitDS_Ts_obj)**2 + \
        #              weight*(Ts_lv1_vector[1] - SS_Ts_obj)**2 + \
        #              weight*(Ts_lv1_vector[2] - DS_Ts_obj)**2

        # #For no timing local obj or Timing fixed or slack constrained
        J_localobj = weight*(x_T    - x_obj)**2    + weight*(y_T    - y_obj)**2    + weight*(z_T    - z_obj)**2    + \
                     weight*(xdot_T - xdot_obj)**2 + weight*(ydot_T - ydot_obj)**2 + weight*(zdot_T - zdot_obj)**2 + \
                     weight*(Lx_T   - Lx_obj)**2   + weight*(Ly_T   - Ly_obj)**2   + weight*(Lz_T   - Lz_obj)**2 + \
                     weight*(Px_lv1 - Px_obj)**2   + weight*(Py_lv1 - Py_obj)**2   + weight*(Pz_lv1 - Pz_obj)**2
    elif ((TotalNumSteps == 1) and (LocalObjTrackingType == None)) or ((TotalNumSteps > 1) and (LocalObjTrackingType == None)): 
        #No matter single step NLP or multi-step NLP, as long as we have None for LocalObjTracking, we give a terminal cost
        J_terminal = 10.0*(x_T - x_end)**2 + 10.0*(y_T - y_end)**2 #+ 10*(z_T - z_end)**2 
    elif (TotalNumSteps > 1) and (LocalObjTrackingType != None):
        raise Exception("Error::Stop here! Detected Local Obj Tracking in Multi-step NLP")

    #Sum up all the cost
    J = J + J_lv1 + J_lv2 + J_localobj + J_terminal

    #-------------------------
    #Add-on Constraints (Local Obj Tracking Constraint and Connection Constraints for Multi-step NLP)
    #   Build Container
    gLocalobj = [];  gLocalobj_lb = [];     gLocalobj_ub = []
    gConnect = [];   gConnect_lb = [];   gConnect_ub = []

    if TotalNumSteps == 1 and LocalObjTrackingType == "constraints":#Single Step NLP with puting Local obj tracking as constraints
        var_lv1_objState = ca.vertcat(var_lv1[var_idx_lv1["x"][-1]],       var_lv1[var_idx_lv1["y"][-1]],      var_lv1[var_idx_lv1["z"][-1]],
                                      var_lv1[var_idx_lv1["xdot"][-1]],    var_lv1[var_idx_lv1["ydot"][-1]],   var_lv1[var_idx_lv1["zdot"][-1]],
                                      var_lv1[var_idx_lv1["Lx"][-1]],      var_lv1[var_idx_lv1["Ly"][-1]],     var_lv1[var_idx_lv1["Lz"][-1]],
                                      var_lv1[var_idx_lv1["px"][-1]],      var_lv1[var_idx_lv1["py"][-1]],     var_lv1[var_idx_lv1["pz"][-1]])
        
        var_localobj = ca.vertcat(x_obj,    y_obj,      z_obj,
                                 xdot_obj,  ydot_obj,   zdot_obj,
                                 Lx_obj,    Ly_obj,     Lz_obj,
                                 Px_obj,    Py_obj,     Pz_obj)
        
        gLocalobj, gLocalobj_lb, gLocalobj_ub = std_eq_constraint(a = var_lv1_objState, b = var_localobj, g = gLocalobj, glb= gLocalobj_lb, gub = gLocalobj_ub)
  
    elif TotalNumSteps > 1: #Only for Multi-Step NLP/Ponton
        #Collects States/Vars
        if SecondLevel == "NLP_SecondLevel":
            #   State/Var for the intitial values of Level 2 (Full States)
            var_lv2_0   = ca.vertcat(var_lv2[var_idx_lv2["x"][0]],       var_lv2[var_idx_lv2["y"][0]],       var_lv2[var_idx_lv2["z"][0]],
                                    var_lv2[var_idx_lv2["xdot"][0]],    var_lv2[var_idx_lv2["ydot"][0]],    var_lv2[var_idx_lv2["zdot"][0]],
                                    var_lv2[var_idx_lv2["Lx"][0]],      var_lv2[var_idx_lv2["Ly"][0]],      var_lv2[var_idx_lv2["Lz"][0]],
                                    var_lv2[var_idx_lv2["Ldotx"][0]],   var_lv2[var_idx_lv2["Ldoty"][0]],   var_lv2[var_idx_lv2["Ldotz"][0]],
                                    var_lv2[var_idx_lv2["px_init"][0]], var_lv2[var_idx_lv2["py_init"][0]], var_lv2[var_idx_lv2["pz_init"][0]])
            #   State/Var for the terminal values of Level 1
            var_lv1_T = ca.vertcat(var_lv1[var_idx_lv1["x"][-1]],       var_lv1[var_idx_lv1["y"][-1]],      var_lv1[var_idx_lv1["z"][-1]],
                                var_lv1[var_idx_lv1["xdot"][-1]],    var_lv1[var_idx_lv1["ydot"][-1]],   var_lv1[var_idx_lv1["zdot"][-1]],
                                var_lv1[var_idx_lv1["Lx"][-1]],      var_lv1[var_idx_lv1["Ly"][-1]],     var_lv1[var_idx_lv1["Lz"][-1]],
                                var_lv1[var_idx_lv1["Ldotx"][-1]],   var_lv1[var_idx_lv1["Ldoty"][-1]],  var_lv1[var_idx_lv1["Ldotz"][-1]],
                                var_lv1[var_idx_lv1["px"][-1]],      var_lv1[var_idx_lv1["py"][-1]],     var_lv1[var_idx_lv1["pz"][-1]])
        elif SecondLevel == "Ponton_FourPoints" or SecondLevel == "Ponton_SinglePoint":
            #   State/Var for the intitial values of Level 2 (Partial State)
            var_lv2_0   = ca.vertcat(var_lv2[var_idx_lv2["x"][0]],       var_lv2[var_idx_lv2["y"][0]],       var_lv2[var_idx_lv2["z"][0]],
                                     var_lv2[var_idx_lv2["xdot"][0]],    var_lv2[var_idx_lv2["ydot"][0]],    var_lv2[var_idx_lv2["zdot"][0]],
                                     var_lv2[var_idx_lv2["px_init"][0]], var_lv2[var_idx_lv2["py_init"][0]], var_lv2[var_idx_lv2["pz_init"][0]])
            #   State/Var for the terminal values of Level 1
            var_lv1_T = ca.vertcat(var_lv1[var_idx_lv1["x"][-1]],       var_lv1[var_idx_lv1["y"][-1]],      var_lv1[var_idx_lv1["z"][-1]],
                                   var_lv1[var_idx_lv1["xdot"][-1]],    var_lv1[var_idx_lv1["ydot"][-1]],   var_lv1[var_idx_lv1["zdot"][-1]],
                                   var_lv1[var_idx_lv1["px"][-1]],      var_lv1[var_idx_lv1["py"][-1]],     var_lv1[var_idx_lv1["pz"][-1]])
        #Enforce equality constraint
        gConnect, gConnect_lb, gConnect_ub = std_eq_constraint(a = var_lv2_0, b = var_lv1_T, g = gConnect, glb= gConnect_lb, gub = gConnect_ub)

    #Collect all Constraints
    g = ca.vertcat(*g_lv1,*g_lv2,*gLocalobj,*gConnect)
    #print(g[6270])

    #Collect all Constraints lower and bounds
    glb = np.concatenate((*glb_lv1, *glb_lv2, *gLocalobj_lb, *gConnect_lb), axis = None)
    gub = np.concatenate((*gub_lv1, *gub_lv2, *gLocalobj_ub, *gConnect_ub), axis = None)

    #-----------------------------------------------------------------------------------------------------------------------
    #   Build Solvers
    prob = {'x': DecisionVars, 'f': J, 'g': g, 'p': paras}
    opts = {}
    
    #------------Kintro------------------
    #Good Setup of Knitro
    opts["knitro.presolve"] = 1
    opts["knitro.honorbnds"] = 0
    opts["knitro.OutLev"] = 2
    opts["knitro.bar_directinterval"] = 0
    opts["knitro.maxit"]=10000
    opts["knitro.maxtime_real"]=max_compute_time
    #opts["knitro.bar_feasible"]=2
    solver = ca.nlpsol('solver', 'knitro', prob, opts)

    #-------------IPOPT------------
    #opts["ipopt.bound_push"] = 1e-7
    #opts["ipopt.bound_frac"] = 1e-7
    #opts["ipopt.constr_viol_tol"] = 1e-3
    #solver = ca.nlpsol('solver', 'ipopt', prob, opts)

    return copy.deepcopy(solver), copy.deepcopy(DecisionVars_lb), copy.deepcopy(DecisionVars_ub), copy.deepcopy(glb), copy.deepcopy(gub), copy.deepcopy(var_index)