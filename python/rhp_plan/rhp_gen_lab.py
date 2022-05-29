#NOTE: !!!When planning from Unseen state, the swing leg flag is also important for the first initial state. (For unseen state, we may swing from left, but we may also swing from right) 

import numpy as np
from multicontact_learning_local_objectives.python.ocp_build import *
from multicontact_learning_local_objectives.python.terrain_create import *
from multicontact_learning_local_objectives.python.rhp_plan.rhp_utils import *
import multicontact_learning_local_objectives.python.visualization as viz
from multicontact_learning_local_objectives.python.terrain_create import *
from multicontact_learning_local_objectives.python.rhp_plan.get_localobj import *

import sys
import pickle
import time
import copy
import os

import rospy
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from centroidal_planning_msgs.msg import MotionPlanData, CoMStateFeedback, FootStateFeedback, FrameTransformation

#-------------------------------
#Reseed the random generator
np.random.seed()

#-------------------------------
#Collect External Parameters
#   Get Useful External Parameter List
externalParasList = sys.argv[1:]
#   Make the default dict
ExternalParameters = {"WorkingDirectory": None,
                      "InitConditionType": "null",
                      "InitConditionFilePath": None, #From computing from stored Unseen InitCondition (for the first step)
                      "RollOutFolderName": "TempRollOuts",
                      "Exp_Prefix": '',
                      "EnvModelPath": None, #Used when doing local obj tracking exp
                      "LocalObjTrackingFlag": "No",
                      "LocalObj_from":"NeuralNetwork",
                      "ML_ModelPath": None, #"/home/jiayu/Desktop/MLP_DataSet/2stepsVsOthers/ML_Models/NN_Model_Valid",
                      "DataSetPath": None, #"/home/jiayu/Desktop/MLP_DataSet/Rubbles/DataSet", #None,
                      "NumLookAhead": 4,
                      "NumofRounds":5,
                      "LargeSlopeAngle": 0,
                      "NoisyLocalObj": "No",
                      "NoiseLevel":0.0, #Noise Level in meters,
                      "VisualizationFlag": "Yes",
                      "TrackTiming": "No"
                      }

#   Update External Parameters
for i in range(len(externalParasList)//2):
    #First check if it is the key of the default the dict
    if externalParasList[2*i][1:] in ExternalParameters: #remove the "-" from the parameter list
        ExternalParameters[externalParasList[2*i][1:]] = externalParasList[2*i + 1] #remove the "-" from the parameter list
    else:
        raise Exception("Unknown Parameter: ", externalParasList[2*i][1:])

#   Convert TrackTiming Flag into boolean
if ExternalParameters["TrackTiming"] == "No":
    ExternalParameters["TrackTiming"] = False
elif ExternalParameters["TrackTiming"] == "Yes":
    ExternalParameters["TrackTiming"] = True
else:
    raise Exception("Unknow TrackTiming Flag")

#-------------------------------
#Log File and Save Data
saveData = True
#Decide save path and name
#   No external parameters
if ExternalParameters["WorkingDirectory"] == None: 
    workingDirectory = "/home/jiayu/Desktop/MLP_DataSet"
    rolloutDirectory = workingDirectory + '/' + ExternalParameters["RollOutFolderName"] #by default, rollout saving directory is the same as working directory
    Filename =  ExternalParameters["Exp_Prefix"] + "temp"+str(time.time())
else:
    workingDirectory = ExternalParameters["WorkingDirectory"]
    rolloutDirectory = workingDirectory + '/' + ExternalParameters["RollOutFolderName"]
    Filename = ExternalParameters["Exp_Prefix"] + "_temp"+str(time.time())

#Check if Directory Exist, if no, then we create
if not (os.path.isdir(rolloutDirectory)):
    os.mkdir(rolloutDirectory)
#Logging command line output
stdoutOrigin=sys.stdout; sys.stdout = open(rolloutDirectory + '/' + Filename + ".txt", "w") if saveData == True else None

#---------------------
#   Some Global Settings
np.set_printoptions(precision=4) #Decide Numerical Printing Precision
if ExternalParameters["VisualizationFlag"] == "Yes":
    showResult = True                #Display Optimization Result
elif ExternalParameters["VisualizationFlag"] == "No":
    showResult = False                #Display Optimization Result
else:
    raise Exception("Unknown Flag")
printResult = True               #Print Trajectories

#-----------------------------------
#Parameters for Problem Set up
#------------------------------------
#   Number of Lookahead (1 = Single Step NLP)
if ExternalParameters["LocalObjTrackingFlag"] == "No":
    NumLookAhead = int(ExternalParameters["NumLookAhead"])#2
elif ExternalParameters["LocalObjTrackingFlag"] == "Yes":
    NumLookAhead = 1
#   Number of Rounds/Steps
Nrounds = int(ExternalParameters["NumofRounds"])#29
#   Number of Knots per phase
N_knots_per_phase = 8#8
#   Robot Mass
RobotMass= 100.0
#   Phase Duration Limits
#To Overcome very very large slopes, the ratio of DS and SS matters and the z-height to footstep distance needs to be lowered
#i.e. DS upper limit 1.0, SS upper limit 1.5
#phase_duration_limits = {"DoubleSupport_Min": 0.05, "DoubleSupport_Max": 1.0, #1.5
#                         "SingleSupport_Min": 0.7,  "SingleSupport_Max": 1.0}
#The one we use
phase_duration_limits = {"DoubleSupport_Min": 0.5, "DoubleSupport_Max": 1.0, #0.05 - 0.5  0.5 - 1.0
                         "SingleSupport_Min": 1.0,  "SingleSupport_Max": 1.5}  #0.7 - 1.2
# phase_duration_limits = {"DoubleSupport_Min": 0.1, "DoubleSupport_Max": 0.1, #0.05 - 0.5
#                          "SingleSupport_Min": 0.8,  "SingleSupport_Max": 0.8}  #0.7 - 1.2
#   Local Obj Tracking Type (for Single Step) can be None, cost, constraints
LocalObjSettings = {}
if NumLookAhead == 1: #Single-Step NLP, there is a point to set up local obj tracking parameters
    
    #NOTE: (Mannual Select) Local Obj Tracking Type, can be : None (No Local Obj Tracking); (as) constraints, (as) cost
    LocalObjSettings["local_obj_tracking_type"] = "cost"
    
    #If we choose to track Local obj
    if LocalObjSettings["local_obj_tracking_type"] != None:
        
        #NOTE: (Mannual Select) Decid where the local obj comes from: 1) None(all zeros), 2) fromFile 3) NeuralNetwork
        LocalObjSettings["local_obj_source"] = ExternalParameters["LocalObj_from"] #"NeuralNetwork"

        #-----Settings for local obj fromFile
        if LocalObjSettings["local_obj_source"] == "fromFile": 
            #NOTE (Mannual Provide): Local Obj trakcing from File (RefTraj), but where the file comes from
            LocalObjSettings["GroundTruthTraj"] = ExternalParameters["EnvModelPath"] #NOTE: The Env Model has trajs as well
            #NOTE (Mannual Provide): Shift World Frame for Local obj: None, InitCoM, InitSurfBorder
            LocalObjSettings["local_obj_world_frame_shift_mode"] = "StanceFoot" #"StanceFoot"
            #NOTE (Mannual Provide): 1) 3DPoins 2) ConvexCombination (weighted sum) 3) FollowRectangelBorder
            LocalObjSettings["contact_representation_type"] = "FollowRectangelBorder" #"FollowRectangelBorder"
            #Scaling Factor --- Unused
            LocalObjSettings["ScalingFactor"] = 1

        #-----Settings for local obj predicted from Nueral Network
        elif LocalObjSettings["local_obj_source"] == "NeuralNetwork":
            #NOTE (Mannual Provide): The Path to store the model
            LocalObjSettings["MLModelPath"] = ExternalParameters["ML_ModelPath"]#"/home/jiayu/Desktop/MLP_DataSet/Flat_and_OneLargeSlope/ML_Models/NN_Model_Fulldata_Validated"
            #NOTE (Mannual Provide) Ground Truth Traj, Default can be None
            LocalObjSettings["GroundTruthTraj"] = ExternalParameters["EnvModelPath"] #NOTE: The Env Model has trajs as well
            #Load corresponding settings
            dataseSettings = pickle.load(open(ExternalParameters["ML_ModelPath"]+"/datasetSettings.p","rb"))
            LocalObjSettings["local_obj_world_frame_shift_mode"] = dataseSettings["Shift_World_Frame_Type"]
            LocalObjSettings["contact_representation_type"] = dataseSettings["Contact_Representation_Type"]
            LocalObjSettings["ScalingFactor"] = dataseSettings["VectorScaleFactor"]
            LocalObjSettings["NumPreviewSteps"] = dataseSettings["NumPreviewSteps"]

        elif LocalObjSettings["local_obj_source"] == "kNN":
            #NOTE (Mannual Provide) Ground Truth Traj, Default can be None
            LocalObjSettings["GroundTruthTraj"] = ExternalParameters["EnvModelPath"] #NOTE: The Env Model has trajs as well
            #Load corresponding settings
            dataseSettings = pickle.load(open(ExternalParameters["DataSetPath"]+"/data.p","rb"))
            LocalObjSettings["local_obj_world_frame_shift_mode"] = dataseSettings["Shift_World_Frame_Type"]
            LocalObjSettings["contact_representation_type"] = dataseSettings["Contact_Representation_Type"]
            LocalObjSettings["ScalingFactor"] = dataseSettings["VectorScaleFactor"]
            LocalObjSettings["NumPreviewSteps"] = dataseSettings["NumPreviewSteps"]

elif NumLookAhead > 1: #Multi Step NLP, all local obj related parameter becomes None
    LocalObjSettings["local_obj_tracking_type"] = None

#------------------
#   Define the Which foot to Swing for the first step; NOTE: For Now, Let us always start from Left Swing foot
#SwingLeftFirst = 1;   SwingRightFirst = 0
#------------------

#   Initial Seed Type 1) random seed 2)from previous result
InitSeedType = "previous"

#---------------------
#Get Current Robot State
#---------------------
rospy.init_node('listener', anonymous=True)
msg_com = rospy.wait_for_message("/biped_walking_dcm_controller/com_states", CoMStateFeedback)
msg_foot = rospy.wait_for_message("/biped_walking_dcm_controller/foot_poses", FootStateFeedback)
msg_frame_transformation = rospy.wait_for_message("/biped_walking_dcm_controller/frame_transformation_odom_to_map", FrameTransformation)
msg_base_odom = rospy.wait_for_message("/biped_walking_dcm_controller/odometry", Odometry)
msg_base_map = rospy.wait_for_message("/biped_walking_dcm_controller/map", Odometry)

#get transformation from odom to map
rotation_odom_to_map = np.empty([3,3])
rotation_odom_to_map[0,:] = np.array([msg_frame_transformation.rot_matrix[0:3]])
rotation_odom_to_map[1,:] = np.array([msg_frame_transformation.rot_matrix[3:6]])
rotation_odom_to_map[2,:] = np.array([msg_frame_transformation.rot_matrix[6:]])

translation_odom_to_map = np.array([msg_frame_transformation.translation_vector])

print('Current CoM pos in (Map) is ', msg_com.actual_com_pos_x_map, msg_com.actual_com_pos_y_map, msg_com.actual_com_pos_z_map)
print('Current Base pos in (Map) is ', msg_base_map.pose.pose.position.x, msg_base_map.pose.pose.position.y, msg_base_map.pose.pose.position.z)
print('Current Left Foot Step Location in (Map) is: ', msg_foot.actual_lf_pos_x_map, msg_foot.actual_lf_pos_y_map, msg_foot.actual_lf_pos_z_map)
print('Current Right Foot Step Location in (Map) is: ', msg_foot.actual_rf_pos_x_map, msg_foot.actual_rf_pos_y_map, msg_foot.actual_rf_pos_z_map)

print('Current CoM pos in (Odom) is ', msg_com.actual_com_pos_x_odom, msg_com.actual_com_pos_y_odom, msg_com.actual_com_pos_z_odom)
print('Current Base pos in (Odom) is ', msg_base_odom.pose.pose.position.x, msg_base_odom.pose.pose.position.y, msg_base_odom.pose.pose.position.z)
print('Current Left Foot Step Location in (Odom) is: ', msg_foot.actual_lf_pos_x_odom, msg_foot.actual_lf_pos_y_odom, msg_foot.actual_lf_pos_z_odom)
print('Current Right Foot Step Location in (Odom) is: ', msg_foot.actual_rf_pos_x_odom, msg_foot.actual_rf_pos_y_odom, msg_foot.actual_rf_pos_z_odom)

print('Translation from Odom to Map: ', translation_odom_to_map)
print('Rotation from Odom to Map: \n', rotation_odom_to_map)

OdomConfig = {}
OdomConfig["CoM_x"] = msg_com.actual_com_pos_x_odom; OdomConfig["CoM_y"] = msg_com.actual_com_pos_y_odom; 
OdomConfig["CoM_z"] = msg_com.actual_com_pos_z_odom

OdomConfig["PLx"] = msg_foot.actual_lf_pos_x_odom; OdomConfig["PLy"] = msg_foot.actual_lf_pos_y_odom; 
OdomConfig["PLz"] = msg_foot.actual_lf_pos_z_odom

OdomConfig["PRx"] = msg_foot.actual_rf_pos_x_odom; OdomConfig["PRy"] = msg_foot.actual_rf_pos_y_odom; 
OdomConfig["PRz"] = msg_foot.actual_rf_pos_z_odom

#---------------------
#Get Environment Model
#---------------------

#Define if we load terrain from file NOTE: None means no, then we generate terrain from code, depends on if we update external parameters
TerrainModelPath = None
if ExternalParameters["InitConditionType"]=="fromFile" or ExternalParameters["InitConditionType"]=="fromFirstRoundTraj":
    TerrainModelPath = ExternalParameters["InitConditionFilePath"]
if ExternalParameters["EnvModelPath"] != None:
    TerrainModelPath = ExternalParameters["EnvModelPath"]

#Define if we use specially designed terrains
SpecialTerrain = False

if TerrainModelPath == None:
    #Terrain with Small (random) Patches
    if SpecialTerrain == False:
        # # #-----------------------------------------
        # #For local Testing
        TerrainSettings = {"terrain_type": "random",#"antfarm_left",
                           "backward_motion": False,
                           "fixed_inclination": None,#0.0/180*np.pi, #radius, None means random inclination
                            "lab_blocks": True,
                            "lab_block_z_shift": 0.006,
                           "random_init_surf_size": False,
                           "random_surfsize_flag": False,
                           "random_Horizontal_Move": False,
                           "MisMatch_Alignment_of_FirstTwoPatches": False, #bool(np.random.choice([True,False],1)), 
                           "MisAligned_Column": None, #can be "left", "right", None (choose randomly)
                           "Projected_Length": 0.4, "Projected_Width": 0.4, #0.55 and 1.0
                           "large_slope_flag":False,
                           "large_slope_index": [],#[np.random.choice([16,17])],#select a patch from number 16 or 17
                           "large_slope_directions": [],#[np.random.choice(["X_positive", "X_negative", "Y_positive", "Y_negative"])], 
                           "large_slope_inclinations": [],#[23/180*np.pi],#[np.round(np.random.uniform(17.0/180*np.pi,25.0/180*np.pi),3)], #if no elevation change, 22 degress is the limit
                           "large_slope_X_shifts": [],#[0.0], 
                           "large_slope_Y_shifts": [],#[0.0], 
                           "large_slope_Z_shifts": [],#[np.random.uniform(-0.25,0.25)],
                           }

        # #--------------------------------------
        # #For Generating Data Point On Uni Server
        # #NOTE: To close large slope, make lists to become [], currently the large slope flat is not functioning
        # TerrainSettings = {"terrain_type": "random",#"antfarm_left",
        #                   "fixed_inclination": None, #0.0/180*np.pi, #radius, None means random inclination
        #                   "random_init_surf_size": False,
        #                   "random_surfsize_flag": False,
        #                   "random_Horizontal_Move": False,
        #                   "MisMatch_Alignment_of_FirstTwoPatches": False,#bool(np.random.choice([True,False],1)), 
        #                   "MisAligned_Column": None, #can be "left", "right", None (choose randomly)
        #                   "Projected_Length": 0.575, "Projected_Width": 1.0,  #length 0.6 with misalignment maybe betters
        #                   "large_slope_flag":False,
        #                   "large_slope_index": [], #[np.random.choice([8,9])],#select a patch from number 16 or 17
        #                   "large_slope_directions": [], #[np.random.choice(["X_positive", "X_negative", "Y_positive", "Y_negative"])], 
        #                   "large_slope_inclinations": [], #[np.round(np.random.uniform(21.0/180*np.pi,30.0/180*np.pi),3)], #if no elevation change, 22 degress is the limit
        #                   "large_slope_X_shifts": [0.0], 
        #                   "large_slope_Y_shifts": [0.0], 
        #                   "large_slope_Z_shifts": [0.0],#[np.random.uniform(-0.25,0.25)],
        #                 }

        # #--------------------------------------
        # #For Generating Data Point On Uni Server
        # #NOTE: To close large slope, make lists to become [], currently the large slope flat is not functioning
        # TerrainSettings = {"terrain_type": "random",#"antfarm_left",
        #                   "fixed_inclination": None, #0.0/180*np.pi, #radius, None means random inclination
        #                   "random_init_surf_size": False,
        #                   "random_surfsize_flag": False,
        #                   "random_Horizontal_Move": False,
        #                   "MisMatch_Alignment_of_FirstTwoPatches": False,#bool(np.random.choice([True,False],1)), 
        #                   "MisAligned_Column": None, #can be "left", "right", None (choose randomly)
        #                   "Projected_Length": 0.575, "Projected_Width": 1.0,  #length 0.6 with misalignment maybe betters
        #                   "large_slope_flag":False,
        #                   "large_slope_index": [8], #[np.random.choice([8,9])],#select a patch from number 16 or 17
        #                   "large_slope_directions": ['Y_positive'], #[np.random.choice(["X_positive", "X_negative", "Y_positive", "Y_negative"])], 
        #                   "large_slope_inclinations": [25.0/180*np.pi], #[np.round(np.random.uniform(21.0/180*np.pi,30.0/180*np.pi),3)], #if no elevation change, 22 degress is the limit
        #                   "large_slope_X_shifts": [0.0], 
        #                   "large_slope_Y_shifts": [0.0], 
        #                   "large_slope_Z_shifts": [0.0],#[np.random.uniform(-0.25,0.25)],
        #                 }

        #Generate Terrain
        TerrainInfo = terrain_model_gen_lab(terrain_name    = TerrainSettings["terrain_type"],          fixed_inclination = TerrainSettings["fixed_inclination"], 
                                            lab_blocks = TerrainSettings["lab_blocks"],
                                            lab_block_z_shift = TerrainSettings["lab_block_z_shift"],
                                            randomInitSurfSize = TerrainSettings["random_init_surf_size"], #False,
                                            random_surfsize = TerrainSettings["random_surfsize_flag"],
                                            randomHorizontalMove = TerrainSettings["random_Horizontal_Move"],
                                            randomMisAlignmentofFirstTwoPatches = TerrainSettings["MisMatch_Alignment_of_FirstTwoPatches"], 
                                            MisAlignmentColumn = TerrainSettings["MisAligned_Column"], 
                                            Proj_Length = TerrainSettings["Projected_Length"], Proj_Width = TerrainSettings["Projected_Width"],
                                            NumSteps = Nrounds, NumLookAhead = 100,#Put NumLookAhead = 20 to give infinitely long terrains
                                            large_slope_flag = TerrainSettings["large_slope_flag"], 
                                            large_slope_index = TerrainSettings["large_slope_index"], large_slope_directions = TerrainSettings["large_slope_directions"], 
                                            large_slope_inclinations = TerrainSettings["large_slope_inclinations"],
                                            large_slope_X_shifts = TerrainSettings["large_slope_X_shifts"], 
                                            large_slope_Y_shifts = TerrainSettings["large_slope_Y_shifts"],
                                            large_slope_Z_shifts = TerrainSettings["large_slope_Z_shifts"]) 
    elif SpecialTerrain == True:
        #Terrain with Specific Patterns (flat/darpa)
        TerrainSettings = {"terrain_type": "darpa_left",#"single_large_slope_far",
                           "fixed_inclination": None, #radius, None means random inclination
                           "random_surfsize_flag": False,
                           "random_Horizontal_Move":False}
        #Create Terrain Model
        TerrainInfo = pre_designed_terrain(terrain_name = TerrainSettings["terrain_type"], NumSteps = Nrounds, NumLookAhead = NumLookAhead)

else:
    print("Load Terrain Model from File: "+TerrainModelPath)
    #Load Terrain Model from file
    with open(TerrainModelPath, 'rb') as f:
            LoadTerrainModel= pickle.load(f)
    TerrainInfo = LoadTerrainModel["TerrainInfo"]
    TerrainSettings = LoadTerrainModel["TerrainSettings"]

#---------------------------------------
#   print a summary for Problem Setups
print("-------------------------------------")
print("Summary for Problem Setups: ")
print("- Number of Lookahead: ", NumLookAhead, " Steps")
print("- Plan ", Nrounds, " Rounds/Steps")
print("- Phase Duration Limits: ")
print("   -Double Support Min: " + str(phase_duration_limits["DoubleSupport_Min"]) + ";   Double Support Max:" + str(phase_duration_limits["DoubleSupport_Max"]))
print("   -Single Support Min: " + str(phase_duration_limits["SingleSupport_Min"]) + ";   Single Support Max:" + str(phase_duration_limits["SingleSupport_Max"]))
if NumLookAhead == 1: #Single Step NLP
    if LocalObjSettings["local_obj_tracking_type"] != None:
        print("- Single Step NLP with Local Obj Tracking with Type of: ", LocalObjSettings["local_obj_tracking_type"])
        print("- Local Obj Got from Source: ", LocalObjSettings["local_obj_source"])
        if LocalObjSettings["local_obj_source"] != None: #Not random local obj
            print("- Local Obj World Frame Shifted as: ", LocalObjSettings["local_obj_world_frame_shift_mode"])
            print("- Local Obj Contact Local Representation Type: ", LocalObjSettings["contact_representation_type"])
            print("- Scaling Factor of (Input/Output Vectors): ", LocalObjSettings["ScalingFactor"])
            print("- Number of Preview Steps in DataSet: ", LocalObjSettings["NumPreviewSteps"]) if (LocalObjSettings["local_obj_source"] == "NeuralNetwork" or LocalObjSettings["local_obj_source"] == "kNN") else None
    elif LocalObjSettings["local_obj_tracking_type"] == None:
        print("- Single Step NLP with Reaching the far goal, (Local Obj Tracking Type: ", LocalObjSettings["local_obj_tracking_type"], "), Will fail for sure")
elif NumLookAhead > 1: #Multi-step NLP
    if LocalObjSettings["local_obj_tracking_type"] != None:
        raise Exception("Error::Local Obj Tracking Enabled during Multi-Step NLP")
    elif LocalObjSettings["local_obj_tracking_type"] == None:
        print("- Multi-Step NLP with ", str(NumLookAhead), " Steps Lookahead")
print("- Initial Seed Type: ", InitSeedType)
print(" ")
#   Print Terrain Information
print("-------------------------------------")
print("Terrain Set up")
print("- Terrain Type/Name: ", TerrainSettings["terrain_type"])
if TerrainSettings["fixed_inclination"] != None:
    print("- Fixed Surf Incliation (radius): ", TerrainSettings["fixed_inclination"])
else:
    print("- Random Surf Inclination")

if TerrainSettings["random_surfsize_flag"] == True:
    print("- Random Shrink the (Projected) Surf Size")
else:
    print("- Fixed (Projected) Surf Size")
if TerrainSettings["random_Horizontal_Move"] == True:
    print("- Random Horizontal Move of the (Projected) Surf Size")
else:
    print("- No random horizontal move")

print(" ")
#   Display Terrain
viz.DisplayResults(TerrainModel = TerrainInfo, SingleOptResult = None, AllOptResult = None) if showResult == True else None

#-------------------------------------------
#  Set Initial Condition (here for the first step) 
#-------------------------------------------
#A dict to contain InitConfig
InitConfig = {}
#Set initial configuration type
#init_config_type = "null"#   Init Config type, NULL Conditon for now (NOTE: may need to replace with some function to auto get these from past results)
#Get Initial Conditions
#NOTE: SwingLeftFirst, and SwingRightFirst are different from InitConfig as they identify Which leg swings at the first step
SwingLeftFirst, SwingRightFirst, \
InitConfig["x_init"],   InitConfig["y_init"],  InitConfig["z_init"],  InitConfig["xdot_init"],  InitConfig["ydot_init"],  InitConfig["zdot_init"],  \
InitConfig["Lx_init"],  InitConfig["Ly_init"], InitConfig["Lz_init"], InitConfig["Ldotx_init"], InitConfig["Ldoty_init"], InitConfig["Ldotz_init"], \
InitConfig["PLx_init"], InitConfig["PLy_init"],InitConfig["PLz_init"],    \
InitConfig["PRx_init"], InitConfig["PRy_init"],InitConfig["PRz_init"]\
= getInitCondition_FirstStep(InitConditionType = ExternalParameters["InitConditionType"], 
                             InitConditionFilePath = ExternalParameters["InitConditionFilePath"])

#Update the InitConfig base the current readings from the robot state (In Map Frame)
InitConfig["x_init"] = msg_com.actual_com_pos_x_map #msg_com.data[0]
InitConfig["y_init"] = msg_com.actual_com_pos_y_map #msg_com.data[1]
InitConfig["z_init"] = msg_com.actual_com_pos_z_map #msg_com.data[2]

InitConfig["PLx_init"] = msg_foot.actual_lf_pos_x_map #msg_foot.data[0]
InitConfig["PLy_init"] = msg_foot.actual_lf_pos_y_map #msg_foot.data[1]
InitConfig["PLz_init"] = msg_foot.actual_lf_pos_z_map #msg_foot.data[2]

InitConfig["PRx_init"] = msg_foot.actual_rf_pos_x_map #msg_foot.data[6]
InitConfig["PRy_init"] = msg_foot.actual_rf_pos_y_map #msg_foot.data[7]
InitConfig["PRz_init"] = msg_foot.actual_rf_pos_z_map #msg_foot.data[8]

#Get Init Contact Surfaces (here for the first step) and Orientation from the terrain info
InitConfig["LeftInitSurf"]  = TerrainInfo["InitLeftSurfVertice"]
InitConfig["RightInitSurf"] = TerrainInfo["InitRightSurfVertice"]
#Get Init Contact Tangent and Norms (Here is for the first step now), compute from get getTerrainTagentsNormOrientation function
InitConfig["PL_init_TangentX"] = TerrainInfo["InitLeftSurfTangentX"];    InitConfig["PL_init_TangentY"] = TerrainInfo["InitLeftSurfTangentY"]
InitConfig["PL_init_Norm"]     = TerrainInfo["InitLeftSurfNorm"];        
InitConfig["PR_init_TangentX"] = TerrainInfo["InitRightSurfTangentX"];    InitConfig["PR_init_TangentY"] = TerrainInfo["InitLeftSurfTangentY"]
InitConfig["PR_init_Norm"]     = TerrainInfo["InitRightSurfNorm"];  
#Get Init Contact Surfaces Orientation
InitConfig["LeftInitSurfOrientation"]  = TerrainInfo["InitLeftSurfOrientation"]
InitConfig["RightInitSurfOrientation"] = TerrainInfo["InitRightSurfOrientation"]
#Get Contact Surface Orientation
InitConfig["SurfOrientations"] = TerrainInfo["ContactSurfsOrientation"][0:0 + NumLookAhead - 1 + 1]

#Old code to get tangent and terrain orientation from terrain utils function
#InitConfig["PL_init_TangentX"], InitConfig["PL_init_TangentY"], InitConfig["PL_init_Norm"], InitConfig["LeftInitSurfOrientation"]  = getTerrainTagentsNormOrientation(InitConfig["LeftInitSurf"])
#InitConfig["PR_init_TangentX"], InitConfig["PR_init_TangentY"], InitConfig["PR_init_Norm"], InitConfig["RightInitSurfOrientation"] = getTerrainTagentsNormOrientation(InitConfig["RightInitSurf"])
#NOTE: Check if PL/PR init are in the Left and Right Init Contact Patches
#---------------------------------------------------------
#   Print Initial Condtion
print("-------------------------------------")
print("Initial Configuration (of the First Step): ")
print("- Swing !Left! Foot at the First Step (SwingLeftFirst = ", str(SwingLeftFirst),")") if SwingLeftFirst == 1 else None
print("- Swing !Right! Foot at the First Step (SwingRightFirst = ", str(SwingRightFirst),")") if SwingRightFirst == 1 else None
print("- Initial Condition Type: ", ExternalParameters["InitConditionType"])
print("- Initial CoM Position: x = ", str(InitConfig["x_init"]), ", y = ", str(InitConfig["y_init"]), ", z = ", str(InitConfig["z_init"]))
print("- Initial CoM Velocity: xdot = ", str(InitConfig["xdot_init"]), ", ydot = ", str(InitConfig["ydot_init"]), ", zdot = ", str(InitConfig["zdot_init"]))
print("- Initial Angular Momentum: Lx = ", str(InitConfig["Lx_init"]), ", Ly = ", str(InitConfig["Ly_init"]), ", Lz = ", str(InitConfig["Lz_init"]))
print("- Initial Angular Momentum rate: Ldotx = ", str(InitConfig["Ldotx_init"]), ", Ldoty = ", str(InitConfig["Ldoty_init"]), ", Ldotz = ", str(InitConfig["Ldotz_init"]), ", AM rate Init is Not Used")
print("- Initial Left Foot Contact Location: PLx_init = ",  str(InitConfig["PLx_init"]), ", PLy_init = ", str(InitConfig["PLy_init"]), ", PLz_init = ", str(InitConfig["PLz_init"]))
print("- Initial Right Foot Contact Location: PRx_init = ", str(InitConfig["PRx_init"]), ", PRy_init = ", str(InitConfig["PRy_init"]), ", PRz_init = ", str(InitConfig["PRz_init"]))
print("- Initial Left Foot: Tangent X = ", InitConfig["PL_init_TangentX"], ",  Tangent Y = ", InitConfig["PL_init_TangentY"], ", Norm = ", InitConfig["PL_init_Norm"])
print("- Initial Left Contact Surface Orientation: \n",  InitConfig["LeftInitSurfOrientation"])
print("- Initial Right Foot: Tangent X = ", InitConfig["PR_init_TangentX"], ", Tangent Y = ", InitConfig["PR_init_TangentY"], ", Norm = ", InitConfig["PR_init_Norm"])
print("- Initial Right Contact Surface Orientation: \n", InitConfig["RightInitSurfOrientation"])

#Display Init config
viz.DisplayInitConfig(TerrainModel=TerrainInfo, InitConfig = InitConfig)
#Display Odom config
viz.DisplayOdomConfig(OdomConfig = OdomConfig)

#----------------------------------------------------------
#   Set (Far) Terminal/Goal State
#----------------------------------------------------------
GoalState = {}
#   Goal/Terminal CoM x, y, z
GoalState["x_end"] = 30.0;           GoalState["y_end"] = 0.0;       GoalState["z_end"] = 0.88
#flip target x if we turn on backward motion
if TerrainSettings["backward_motion"] == True:
    GoalState["x_end"] = -GoalState["x_end"]
#   Goal/Terminal CoMdot x, y, z (Not Used for now)
GoalState["xdot_end"] = 0.0;         GoalState["ydot_end"] = 0.0;    GoalState["zdot_end"] = 0.0
#   Print Terminal/goal State
print("Terminal/Goal CoM Position: x = ", str(GoalState["x_end"]), " y = ", str(GoalState["y_end"]), " z = ", str(GoalState["z_end"]), "(z-height un-tracked)\n") if LocalObjSettings["local_obj_tracking_type"] == None else None

#-----------------
#Build Solver
if NumLookAhead == 1: #Single Step NLP
    solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = ocp_solver_build(FirstLevel = "NLP_SingleStep", SecondLevel = None, TotalNumSteps = NumLookAhead, \
                                                                                     LocalObjTrackingType = LocalObjSettings["local_obj_tracking_type"], \
                                                                                     N_knots_local = N_knots_per_phase, robot_mass = RobotMass, \
                                                                                     PhaseDurationLimits=phase_duration_limits,
                                                                                     backward_motion_flag=TerrainSettings["backward_motion"],
                                                                                     TrackingTiming = ExternalParameters["TrackTiming"])
elif NumLookAhead > 1: #Multiple Steps NLP
    solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = ocp_solver_build(FirstLevel = "NLP_SingleStep", SecondLevel = "NLP_SecondLevel", \
                                                                                     TotalNumSteps = NumLookAhead, LocalObjTrackingType = None, \
                                                                                     N_knots_local = N_knots_per_phase, robot_mass = RobotMass,
                                                                                     PhaseDurationLimits=phase_duration_limits,
                                                                                     backward_motion_flag=TerrainSettings["backward_motion"])

    # solver, DecisionVars_lb, DecisionVars_ub, glb, gub, var_index = ocp_solver_build(FirstLevel = "NLP_SingleStep", SecondLevel = "Ponton_SinglePoint", \
    #                                                                                  TotalNumSteps = NumLookAhead, LocalObjTrackingType = None,
    #                                                                                  backward_motion_flag=TerrainSettings["backward_motion"]))

#----------------------------------------
#Start Computing Trajectories
#Make Intial Seed Container first
#   Get DecisionVar Shape #Remake the code
DecisionVarsShape = DecisionVars_lb.shape
#   Make a random initial seed
#   Generate Random Seed from scratch
np.random.seed()
vars_init = DecisionVars_lb + np.multiply(np.random.rand(DecisionVarsShape[0],).flatten(),(DecisionVars_ub - DecisionVars_lb))#   Fixed Value Initial Guess
#   Build an initial seed container; except round/step 0 and 1, container[0] for used when calling the current solver, afterwards container[1] move to container[0] current opt result become container[1]
DecisionVars_init_list = [vars_init, vars_init]

#Define x_opt before use, a small array, if the first round/step (round 0) enquiries, will then report error
x_opt = np.array([1])

#Result Container
AllOptResult = [] #List of Result/Inputs/Paramaters for a Single Optimization
AllTraj = []      #List of raw x_opt (Optimization vector)
CasadiParas = []  #List of casadi parameters

#Computing Trajectories
for roundNum in range(Nrounds):
    print("---------------------------------------")
    print("For The ", roundNum, "Round/Step: ")
    print("---------------------------------------")

    #----------
    #Constructing Casadi Parameters Based on Previous Result or For the First Round
    #   Left and Right Swing flags
    if roundNum%2 == 0: #even number of rounds/steps
        if SwingLeftFirst == 1:#If First step swing the left, then even number of rounds/steps swing the left
            InitConfig["LeftSwingFlag"] = 1;   InitConfig["RightSwingFlag"] = 0
        elif SwingRightFirst == 1:#If the first step Swing the Right, then the even number of rounds/steps swing the right
            InitConfig["LeftSwingFlag"] = 0;   InitConfig["RightSwingFlag"] = 1
    elif roundNum%2 == 1: #odd number of rounds/steps
        if SwingLeftFirst == 1:#If the First step swing the left, then the odd number of rounds/steps swing the right
            InitConfig["LeftSwingFlag"] = 0;   InitConfig["RightSwingFlag"] = 1
        elif SwingRightFirst == 1:#If the first step swing the right, then the odd number of round swing the left
            InitConfig["LeftSwingFlag"] = 1;   InitConfig["RightSwingFlag"] = 0

    #   Update Initial Condition (only for round/step larger than 0 - start from the second level)
    if roundNum > 0: 
        var_idx_lv1 = var_index["Level1_Var_Index"]
        #CoM x, y, z
        InitConfig["x_init"] = x_opt[var_idx_lv1["x"][0]:var_idx_lv1["x"][1]+1][-1]              
        InitConfig["y_init"] = x_opt[var_idx_lv1["y"][0]:var_idx_lv1["y"][1]+1][-1]
        InitConfig["z_init"] = x_opt[var_idx_lv1["z"][0]:var_idx_lv1["z"][1]+1][-1]
        #CoMdot x, y, z
        InitConfig["xdot_init"]  = x_opt[var_idx_lv1["xdot"][0]:var_idx_lv1["xdot"][1]+1][-1]    
        InitConfig["ydot_init"]  = x_opt[var_idx_lv1["ydot"][0]:var_idx_lv1["ydot"][1]+1][-1]
        InitConfig["zdot_init"]  = x_opt[var_idx_lv1["zdot"][0]:var_idx_lv1["zdot"][1]+1][-1]
        #Angular Momentum
        InitConfig["Lx_init"] = x_opt[var_idx_lv1["Lx"][0]:var_idx_lv1["Lx"][1]+1][-1]
        InitConfig["Ly_init"] = x_opt[var_idx_lv1["Ly"][0]:var_idx_lv1["Ly"][1]+1][-1]
        InitConfig["Lz_init"] = x_opt[var_idx_lv1["Lz"][0]:var_idx_lv1["Lz"][1]+1][-1]
        #Angular Momentum Rate
        InitConfig["Ldotx_init"] = x_opt[var_idx_lv1["Ldotx"][0]:var_idx_lv1["Ldotx"][1]+1][-1]  
        InitConfig["Ldoty_init"] = x_opt[var_idx_lv1["Ldoty"][0]:var_idx_lv1["Ldoty"][1]+1][-1]
        InitConfig["Ldotz_init"] = x_opt[var_idx_lv1["Ldotz"][0]:var_idx_lv1["Ldotz"][1]+1][-1]
        #To Update Intial Left and Right Contact Location, Tangents, Norms, firstly gets the landed foot location and tangents X, Y, Norm
        Px_pre = x_opt[var_idx_lv1["px"][0]:var_idx_lv1["px"][1]+1][-1];            Py_pre = x_opt[var_idx_lv1["py"][0]:var_idx_lv1["py"][1]+1][-1]
        Pz_pre = x_opt[var_idx_lv1["pz"][0]:var_idx_lv1["pz"][1]+1][-1]
        P_pre_TangentX = TerrainInfo["ContactSurfsTangentX"][roundNum - 1];         P_pre_TangentY = TerrainInfo["ContactSurfsTangentY"][roundNum - 1]
        P_pre_Norm     = TerrainInfo["ContactSurfsNorm"][roundNum - 1]
        #Assign Contact Location, Tangents, Norms based on Round/Step number and First step Foot Swing Flag, also get initial contact patches (vertex representation)
        if roundNum%2 == 0: #even number of rounds/steps
            if SwingLeftFirst == 1:
                #If First step swing the left, then CURRENT even number of rounds/steps needs to swing the left
                #Then the previous step should swing the right, and we need to update the Right Init Contact for current round/step
                InitConfig["PRx_init"] = Px_pre;                   InitConfig["PRy_init"] = Py_pre;                   InitConfig["PRz_init"] = Pz_pre
                InitConfig["PR_init_TangentX"] = P_pre_TangentX;   InitConfig["PR_init_TangentY"] = P_pre_TangentY;   InitConfig["PR_init_Norm"] = P_pre_Norm
                InitConfig["RightInitSurf"] = TerrainInfo["ContactSurfsVertice"][roundNum-1]
                InitConfig["RightInitSurfOrientation"] = TerrainInfo["ContactSurfsOrientation"][roundNum-1]

            elif SwingRightFirst == 1:
                #If First step swing the Right, then CURRENT even number of rounds/steps needs to swing the Right
                #Then the previous step should swing the Left, and we need to update the Left Init Contact for the current round/step
                InitConfig["PLx_init"] = Px_pre;                   InitConfig["PLy_init"] = Py_pre;                   InitConfig["PLz_init"] = Pz_pre
                InitConfig["PL_init_TangentX"] = P_pre_TangentX;   InitConfig["PL_init_TangentY"] = P_pre_TangentY;   InitConfig["PL_init_Norm"] = P_pre_Norm
                InitConfig["LeftInitSurf"] = TerrainInfo["ContactSurfsVertice"][roundNum-1]
                InitConfig["LeftInitSurfOrientation"] = TerrainInfo["ContactSurfsOrientation"][roundNum-1]

        elif roundNum%2 == 1:#odd number of rounds/steps
            if SwingLeftFirst == 1:
                #If First step swing the left, then CURRENT odd number of rounds/steps needs to swing the right
                #Then the previous step should swing the left, and we need to update the left Init Contact for current round/step
                InitConfig["PLx_init"] = Px_pre;                   InitConfig["PLy_init"] = Py_pre;                   InitConfig["PLz_init"] = Pz_pre
                InitConfig["PL_init_TangentX"] = P_pre_TangentX;   InitConfig["PL_init_TangentY"] = P_pre_TangentY;   InitConfig["PL_init_Norm"] = P_pre_Norm
                InitConfig["LeftInitSurf"] = TerrainInfo["ContactSurfsVertice"][roundNum-1]
                InitConfig["LeftInitSurfOrientation"] = TerrainInfo["ContactSurfsOrientation"][roundNum-1]

            elif SwingRightFirst == 1:
                #If First step swing the Right, then CURRENT odd number of rounds/steps needs to swing the Left
                #Then the previous step should swing the Right, and we need to update the Right Init Contact for the current round/step
                InitConfig["PRx_init"] = Px_pre;                   InitConfig["PRy_init"] = Py_pre;                   InitConfig["PRz_init"] = Pz_pre
                InitConfig["PR_init_TangentX"] = P_pre_TangentX;   InitConfig["PR_init_TangentY"] = P_pre_TangentY;   InitConfig["PR_init_Norm"] = P_pre_Norm
                InitConfig["RightInitSurf"] = TerrainInfo["ContactSurfsVertice"][roundNum-1]
                InitConfig["RightInitSurfOrientation"] = TerrainInfo["ContactSurfsOrientation"][roundNum-1]

    #   Get Terrain Patches
    InitConfig["SurfParas"]        = TerrainInfo["ContactSurfsHalfSpace"][roundNum:roundNum + NumLookAhead - 1 + 1]
    InitConfig["ContactSurfs"]     = TerrainInfo["ContactSurfsVertice"][roundNum:roundNum + NumLookAhead - 1 + 1] #not used for computation but used for plotting
    InitConfig["SurfTangentsX"]    = TerrainInfo["ContactSurfsTangentX"][roundNum:roundNum + NumLookAhead - 1 + 1]
    InitConfig["SurfTangentsY"]    = TerrainInfo["ContactSurfsTangentY"][roundNum:roundNum + NumLookAhead - 1 + 1]
    InitConfig["SurfNorms"]        = TerrainInfo["ContactSurfsNorm"][roundNum:roundNum + NumLookAhead - 1 + 1]
    InitConfig["SurfOrientations"] = TerrainInfo["ContactSurfsOrientation"][roundNum:roundNum + NumLookAhead - 1 + 1]
    #Get Preview Surfs for Local Obj Prediction when using Machine Learning (only valid for Local Obj Prediction)
    #need to have deep copy otherwise pointing to the same terrain object w.r.t "ContactSurfs" 
    InitConfig["PreviewSurfs_for_Prediction"] = copy.deepcopy(TerrainInfo["ContactSurfsVertice"][roundNum:roundNum + LocalObjSettings["NumPreviewSteps"] - 1 + 1]) \
    if (LocalObjSettings["local_obj_tracking_type"] != None) and (LocalObjSettings["local_obj_source"] == "NeuralNetwork" or LocalObjSettings["local_obj_source"] == "kNN") else None

    #-------------
    #   Get Local Obj if we want to have local obj tracking
    #       No Local Obj Tracking, then Local obj all 0
    if LocalObjSettings["local_obj_tracking_type"] == None:
        LocalObj = getLocalobj(Mode = None)
    #       Yes, we need to track local obj
    elif LocalObjSettings["local_obj_tracking_type"] != None:
        #   Get local obj from file (for sanity check/compare)
        if LocalObjSettings["local_obj_source"] == "fromFile":
            LocalObj = getLocalobj(Mode = LocalObjSettings["local_obj_source"], refTrajFile = LocalObjSettings["GroundTruthTraj"], 
                                  shift_world_frame = LocalObjSettings["local_obj_world_frame_shift_mode"],  roundNum = roundNum, 
                                  ContactParameterizationType = LocalObjSettings["contact_representation_type"],
                                  ScaleFactor = LocalObjSettings["ScalingFactor"])
        #   Get Local obj from Neural Network
        elif LocalObjSettings["local_obj_source"] == "NeuralNetwork":
            #Get Local Obj Predictions (Unshifted to World Frame, but Scale to normal Unit)
            LocalObj = getLocalobj(Mode = LocalObjSettings["local_obj_source"], MLModelPath = LocalObjSettings["MLModelPath"],
                                   shift_world_frame = LocalObjSettings["local_obj_world_frame_shift_mode"], ContactParameterizationType = LocalObjSettings["contact_representation_type"],
                                   ScaleFactor = LocalObjSettings["ScalingFactor"], 
                                   InitConfig = InitConfig)
        #   Get Local obj from kNN
        elif LocalObjSettings["local_obj_source"] == "kNN":
            LocalObj = getLocalobj(Mode = LocalObjSettings["local_obj_source"], 
                                    shift_world_frame = LocalObjSettings["local_obj_world_frame_shift_mode"], 
                                    ContactParameterizationType = LocalObjSettings["contact_representation_type"],
                                    ScaleFactor = LocalObjSettings["ScalingFactor"], 
                                    InitConfig = InitConfig,
                                    DataSetPath = ExternalParameters["DataSetPath"])

        #After Getting LocalObj, Shifting of Variables NOTE: x,y need to shift all together for Init CoM case but not implemented yet
        LocalObj = shiftLocalObj_to_WorldFrame(InitConfig = InitConfig, LocalObj = LocalObj, Local_Frame_Selection = LocalObjSettings["local_obj_world_frame_shift_mode"])

        #Add noise to Localobj if we want
        if ExternalParameters["NoisyLocalObj"] == "Yes" and float(ExternalParameters["NoiseLevel"]) > 0.0:
            print(" ")
            print("---Add Noise to Local Obj")
            print(" ")
            LocalObj = add_Noise_to_Localobj(LocalObj = LocalObj, InitConfig = InitConfig, noiseLevel = float(ExternalParameters["NoiseLevel"]))

        #------------
        #   Show local obj (Only if we enable Local Obj tracking)
        if showResult == True:
            viz.viewLocalObj(InitConfig = InitConfig, LocalObj = LocalObj, TerrainModel = TerrainInfo, roundNum = roundNum,
                            groundTruthTrajPath = LocalObjSettings["GroundTruthTraj"])

    #-------------
    #   Print Info for Current Round
    #       Try to get GroundTruch Initial Configuration of Current Round    
    if ExternalParameters["LocalObjTrackingFlag"] == "Yes":
        if ExternalParameters["EnvModelPath"] == None:
            groundTruthInitConfig = None; groundTruthTerminalConfig = None
        else:
            groundTruthInitConfig, groundTruthTerminalConfig = getInitConfig_in_GlobalFrame_from_file(FilePath=LocalObjSettings["GroundTruthTraj"], RoundNum=roundNum) 
    elif ExternalParameters["LocalObjTrackingFlag"] == "No":
        groundTruthInitConfig = None; groundTruthTerminalConfig = None
    else:
        raise Exception("Unknown LocalObjTrackingFlag")


    #       Print Current Set up and GroudTruth
    #           Swing Foot Flag
    print("-----Swing Foot Flag-------------")
    print("- Swing Left Foot") if InitConfig["LeftSwingFlag"] == 1 else None;      print("- Swing Right Foot") if InitConfig["RightSwingFlag"] == 1 else None
    if groundTruthInitConfig != None:
        print("- Swing Left Foot (Ground Truth)") if groundTruthInitConfig["LeftSwingFlag"] == 1 else None; print("- Swing Right Foot (Ground Truth)") if groundTruthInitConfig["RightSwingFlag"] == 1 else None
    #           Initial CoM
    print("-----Initial CoM-------------")
    print("- Initial CoM Position: x = ", str(InitConfig["x_init"]), ", y = ", str(InitConfig["y_init"]), ", z = ", str(InitConfig["z_init"]))
    print("- Initial CoM Position (Ground Truth): x = ", str(groundTruthInitConfig["x_init"]), ", y = ", str(groundTruthInitConfig["y_init"]), ", z = ", str(groundTruthInitConfig["z_init"])) if groundTruthInitConfig != None else None
    print("- Initial Com Position Diff (Absolute Value): x = ", str(np.absolute(groundTruthInitConfig["x_init"]-InitConfig["x_init"])), ", y = ", str(np.absolute(groundTruthInitConfig["y_init"]-InitConfig["y_init"])), ", z = ", str(np.absolute(groundTruthInitConfig["z_init"]-InitConfig["z_init"]))) if groundTruthInitConfig != None else None
    #           Initial CoMdot
    print("-----Initial CoM Velocity-------------")
    print("- Initial CoM Velocity: xdot = ", str(InitConfig["xdot_init"]), ", ydot = ", str(InitConfig["ydot_init"]), ", zdot = ", str(InitConfig["zdot_init"]))
    print("- Initial CoM Velocity (Ground Truth): xdot = ", str(groundTruthInitConfig["xdot_init"]), ", ydot = ", str(groundTruthInitConfig["ydot_init"]), ", zdot = ", str(groundTruthInitConfig["zdot_init"])) if groundTruthInitConfig != None else None
    print("- Initial Com Velocity Diff (Absolute Value): xdot = ", str(np.absolute(groundTruthInitConfig["xdot_init"]-InitConfig["xdot_init"])), ", ydot = ", str(np.absolute(groundTruthInitConfig["ydot_init"]-InitConfig["ydot_init"])), ", zdot = ", str(np.absolute(groundTruthInitConfig["zdot_init"]-InitConfig["zdot_init"]))) if groundTruthInitConfig != None else None
    #           Angular Momentum
    print("-----Inital Angular Momentum-------------")
    print("- Initial Angular Momentum: Lx = ", str(InitConfig["Lx_init"]), ", Ly = ", str(InitConfig["Ly_init"]), ", Lz = ", str(InitConfig["Lz_init"]))
    print("- Initial Angular Momentum (Ground Truth): Lx = ", str(groundTruthInitConfig["Lx_init"]), ", Ly = ", str(groundTruthInitConfig["Ly_init"]), ", Lz = ", str(groundTruthInitConfig["Lz_init"])) if groundTruthInitConfig != None else None
    print("- Initial Angular Momentum Diff (Absolute Value): Lx = ", str(np.absolute(groundTruthInitConfig["Lx_init"]-InitConfig["Lx_init"])), ", Ly = ", str(np.absolute(groundTruthInitConfig["Ly_init"]-InitConfig["Ly_init"])), ", Lz = ", str(np.absolute(groundTruthInitConfig["Lz_init"]-InitConfig["Lz_init"]))) if groundTruthInitConfig != None else None
    #           Angular Momentum Rate (Not used for Initial config, not need to compare to ground truth)
    print("-----Initial Angular Momenturm Rate (not used for initial config, no need to compare to ground truth)----------")
    print("- Initial Angular Momentum rate: Ldotx = ", str(InitConfig["Ldotx_init"]), ", Ldoty = ", str(InitConfig["Ldoty_init"]), ", Ldotz = ", str(InitConfig["Ldotz_init"]), ", AM rate Init is Not Used")
    #           Left foot Contact Location
    print("-----Inital Left Foot Contact Location-------------")
    print("- Initial Left Foot Contact Location: PLx_init = ", str(InitConfig["PLx_init"]), ", PLy_init = ", str(InitConfig["PLy_init"]), ", PLz_init = ", str(InitConfig["PLz_init"]))
    print("- Initial Left Foot Contact Location (Ground Truth): PLx_init = ", str(groundTruthInitConfig["PLx_init"]), ", PLy_init = ", str(groundTruthInitConfig["PLy_init"]), ", PLz_init = ", str(groundTruthInitConfig["PLz_init"])) if groundTruthInitConfig != None else None
    print("- Initial Left Foot Contact Location Diff (Absolute Value): PLx_init = ", str(np.absolute(groundTruthInitConfig["PLx_init"]-InitConfig["PLx_init"])), ", PLy_init = ", str(np.absolute(groundTruthInitConfig["PLy_init"]-InitConfig["PLy_init"])), ", PLz_init = ", str(np.absolute(groundTruthInitConfig["PLz_init"]-InitConfig["PLz_init"]))) if groundTruthInitConfig != None else None
    #           Right foot Contact Location
    print("-----Inital Right Foot Contact Location-------------")
    print("- Initial Right Foot Contact Location: PRx_init = ", str(InitConfig["PRx_init"]), ", PRy_init = ", str(InitConfig["PRy_init"]), ", PRz_init = ", str(InitConfig["PRz_init"]))
    print("- Initial Right Foot Contact Location (Ground Truth): PRx_init = ", str(groundTruthInitConfig["PRx_init"]), ", PRy_init = ", str(groundTruthInitConfig["PRy_init"]), ", PRz_init = ", str(groundTruthInitConfig["PRz_init"])) if groundTruthInitConfig != None else None
    print("- Initial Right Foot Contact Location Diff (Absolute Value): PRx_init = ", str(np.absolute(groundTruthInitConfig["PRx_init"]-InitConfig["PRx_init"])), ", PRy_init = ", str(np.absolute(groundTruthInitConfig["PRy_init"]-InitConfig["PRy_init"])), ", PRz_init = ", str(np.absolute(groundTruthInitConfig["PRz_init"]-InitConfig["PRz_init"]))) if groundTruthInitConfig != None else None
    #           Left Foot Contact Tangent and Norm
    print("-----Inital Left Foot Contact Tangent and Norm-------------")
    print("- Initial Left Foot: Tangent X = ", InitConfig["PL_init_TangentX"], ",  Tangent Y = ", InitConfig["PL_init_TangentY"], ", Norm = ", InitConfig["PL_init_Norm"])
    print("- Initial Left Foot (Ground Truth): Tangent X = ", groundTruthInitConfig["PL_init_TangentX"], ",  Tangent Y = ", groundTruthInitConfig["PL_init_TangentY"], ", Norm = ", groundTruthInitConfig["PL_init_Norm"]) if groundTruthInitConfig != None else None
    #           Left Foot Contact Orientation
    print("-----Inital Left Foot Contact Surface Orientation-------------")
    print("- Initial Left Contact Surface Orientation: \n",  InitConfig["LeftInitSurfOrientation"])
    print("- Initial Left Contact Surface Orientation (Ground Truth): \n",  groundTruthInitConfig["LeftInitSurfOrientation"]) if groundTruthInitConfig != None else None
    #           Right Foot Contact Orientation
    print("-----Inital Right Foot Contact Tangent and Norm-------------")
    print("- Initial Right Foot: Tangent X = ", InitConfig["PR_init_TangentX"], ", Tangent Y = ", InitConfig["PR_init_TangentY"], ", Norm = ", InitConfig["PR_init_Norm"])
    print("- Initial Right Foot (Ground Truth): Tangent X = ", groundTruthInitConfig["PR_init_TangentX"], ", Tangent Y = ", groundTruthInitConfig["PR_init_TangentY"], ", Norm = ", groundTruthInitConfig["PR_init_Norm"]) if groundTruthInitConfig != None else None
    #           Right Foot Contact Orientation
    print("-----Inital Right Foot Contact Surface Orientation-------------")
    print("- Initial Right Contact Surface Orientation: \n", InitConfig["RightInitSurfOrientation"])
    print("- Initial Right Contact Surface Orientation (Ground Truth): \n", groundTruthInitConfig["RightInitSurfOrientation"]) if groundTruthInitConfig != None else None
    #           Left Foot Init Contact Surface
    print("-----Inital Left Foot Contact Surface Vertices-------------")
    print("- Initial Left Contact Surface: \n", InitConfig["LeftInitSurf"])
    print("- Initial Left Contact Surface (Ground Truth): \n", groundTruthInitConfig["LeftInitSurf"]) if groundTruthInitConfig != None else None
    #           Right Foot Init Contact Surface
    print("-----Inital Right Foot Contact Surface Vertices-------------")
    print("- Initial Left Contact Surface: \n", InitConfig["RightInitSurf"])
    print("- Initial Left Contact Surface (Ground Truth): \n", groundTruthInitConfig["RightInitSurf"]) if groundTruthInitConfig != None else None

    #---------------------
    #Print Local Obj Info
    if LocalObjSettings["local_obj_tracking_type"] != None:
        #For Local Obj CoM
        print("-----Local Obj CoM Position-------------")
        print("- Local Obj CoM Position (Shifted to World Frame and Scaled back to normal): x = ", str(LocalObj["x_obj"]), ", y = ", str(LocalObj["y_obj"]), ", z = ", str(LocalObj["z_obj"]))
        print("- Local Obj CoM Position (Ground Truth): x = ", str(groundTruthTerminalConfig["x_end"]), "y = ", str(groundTruthTerminalConfig["y_end"]), "z = ", str(groundTruthTerminalConfig["z_end"])) if groundTruthTerminalConfig != None else None
        print("- Local Obj CoM Position Diff (Absolute Value): x = ", str(np.absolute(LocalObj["x_obj"]-groundTruthTerminalConfig["x_end"])), "y = ", str(np.absolute(LocalObj["y_obj"]-groundTruthTerminalConfig["y_end"])), "z = ", str(np.absolute(LocalObj["z_obj"]-groundTruthTerminalConfig["z_end"]))) if groundTruthTerminalConfig != None else None
        #For Local Obj CoMdot
        print("-----Local Obj CoM Velocity-------------")
        print("- Local Obj CoM Velocity (Scaled back to normal): xdot = ", str(LocalObj["xdot_obj"]), ", ydot = ", str(LocalObj["ydot_obj"]), ", zdot = ", str(LocalObj["zdot_obj"]))
        print("- Local Obj CoM Velocity (Ground Truth): xdot = ", str(groundTruthTerminalConfig["xdot_end"]), "ydot = ", str(groundTruthTerminalConfig["ydot_end"]), "zdot = ", str(groundTruthTerminalConfig["zdot_end"])) if groundTruthTerminalConfig != None else None
        print("- Local Obj CoM Velocity Diff (Absolute Value): xdot = ", str(np.absolute(LocalObj["xdot_obj"]-groundTruthTerminalConfig["xdot_end"])), "ydot = ", str(np.absolute(LocalObj["ydot_obj"]-groundTruthTerminalConfig["ydot_end"])), "zdot = ", str(np.absolute(LocalObj["zdot_obj"]-groundTruthTerminalConfig["zdot_end"]))) if groundTruthTerminalConfig != None else None
        #For Local Obj AM
        print("-----Local Obj Angular Momentum-------------")
        print("- Local Obj Angular Momentum (Scaled back to normal): Lx = ", str(LocalObj["Lx_obj"]), ", Ly = ", str(LocalObj["Ly_obj"]), ", Lz = ", str(LocalObj["Lz_obj"]))
        print("- Local Obj Angular Momentum (Ground Truth): Lx = ", str(groundTruthTerminalConfig["Lx_end"]), "Ly = ", str(groundTruthTerminalConfig["Ly_end"]), "Lz = ", str(groundTruthTerminalConfig["Lz_end"])) if groundTruthTerminalConfig != None else None
        print("- Local Obj Angular Momentum Diff (Absolute Value): Lx = ", str(np.absolute(LocalObj["Lx_obj"]-groundTruthTerminalConfig["Lx_end"])), "Ly = ", str(np.absolute(LocalObj["Ly_obj"]-groundTruthTerminalConfig["Ly_end"])), "Lz = ", str(np.absolute(LocalObj["Lz_obj"]-groundTruthTerminalConfig["Lz_end"]))) if groundTruthTerminalConfig != None else None
        #For Local Obj Contact Locations
        print("-----Local Obj Contact Location-------------")
        print("- Local Obj Target Contact Location (Shifted to World Frame and Scaled back to normal)): Px = ", str(LocalObj["Px_obj"]), ", Py = ", str(LocalObj["Py_obj"]), ", Pz = ", str(LocalObj["Pz_obj"]))
        print("- Local Obj Target Contact Location (Ground Truth): Px = ", str(groundTruthTerminalConfig["Px"]), "Py = ", str(groundTruthTerminalConfig["Py"]), "Pz = ", str(groundTruthTerminalConfig["Pz"])) if groundTruthTerminalConfig != None else None
        print("- Local Obj Target Contact Location Diff (Absolute Value): Px = ", str(np.absolute(LocalObj["Px_obj"]-groundTruthTerminalConfig["Px"])), "Py = ", str(np.absolute(LocalObj["Py_obj"]-groundTruthTerminalConfig["Py"])), "Pz = ", str(np.absolute(LocalObj["Pz_obj"]-groundTruthTerminalConfig["Pz"]))) if groundTruthTerminalConfig != None else None
    print(" ")

    #---------------
    #Collect all parameters for casadi
    ParaList = np.concatenate((InitConfig["LeftSwingFlag"], InitConfig["RightSwingFlag"],
                               InitConfig["x_init"],        InitConfig["y_init"],        InitConfig["z_init"],   
                               InitConfig["xdot_init"],     InitConfig["ydot_init"],     InitConfig["zdot_init"],
                               InitConfig["Lx_init"],       InitConfig["Ly_init"],       InitConfig["Lz_init"],  
                               InitConfig["Ldotx_init"],    InitConfig["Ldoty_init"],    InitConfig["Ldotz_init"],
                               InitConfig["PLx_init"],      InitConfig["PLy_init"],      InitConfig["PLz_init"], 
                               InitConfig["PRx_init"],      InitConfig["PRy_init"],      InitConfig["PRz_init"],
                               GoalState["x_end"],          GoalState["y_end"],          GoalState["z_end"],    
                               GoalState["xdot_end"],       GoalState["ydot_end"],       GoalState["zdot_end"],
                               InitConfig["SurfParas"],     InitConfig["SurfTangentsX"], InitConfig["SurfTangentsY"],  InitConfig["SurfNorms"],  InitConfig["SurfOrientations"],
                               InitConfig["PL_init_TangentX"],   InitConfig["PL_init_TangentY"],   InitConfig["PL_init_Norm"],  InitConfig["LeftInitSurfOrientation"],
                               InitConfig["PR_init_TangentX"],   InitConfig["PR_init_TangentY"],   InitConfig["PR_init_Norm"],  InitConfig["RightInitSurfOrientation"],
                               LocalObj["x_obj"],  LocalObj["y_obj"],  LocalObj["z_obj"],  LocalObj["xdot_obj"], LocalObj["ydot_obj"], LocalObj["zdot_obj"],
                               LocalObj["Lx_obj"], LocalObj["Ly_obj"], LocalObj["Lz_obj"], LocalObj["Px_obj"],   LocalObj["Py_obj"],   LocalObj["Pz_obj"],
                               LocalObj["InitDS_Ts_obj"], LocalObj["SS_Ts_obj"], LocalObj["DS_Ts_obj"]), axis = None)
    
    #----------
    #Call solver
    start_time = time.time()
    res = solver(x0=DecisionVars_init_list[0], p = ParaList, lbx = DecisionVars_lb, ubx = DecisionVars_ub, lbg = glb, ubg = gub)
    end_time = time.time()
    time_diff = end_time-start_time
    #get result vector
    x_opt = res["x"].full().flatten()

    #------------
    #Update Initial seed
    if InitSeedType == "random":    #Option 1) random initial seed
        np.random.seed()
        vars_init = DecisionVars_lb + np.multiply(np.random.rand(DecisionVarsShape[0],).flatten(),(DecisionVars_ub - DecisionVars_lb))#   Fixed Value Initial Guess
        DecisionVars_init_list = [vars_init, vars_init]
    elif InitSeedType == "previous": #Option 2) With Initial Seed of Previous Type
    #   Except the 0 and 1 round/step, Everytime we use DecisionVar_init_list[0] as the initial seed,
    #   And everytime we compute the result, DecisionVar_init_list[0] = DecisionVar_init_list[1]. DecisionVar_init_list[1] = new_opt_result
        if roundNum == 0: #The first round
            DecisionVars_init_list = [x_opt, x_opt]
        elif roundNum > 1:
            DecisionVars_init_list[0] = DecisionVars_init_list[1]
            DecisionVars_init_list[1] = x_opt
    else:
        raise Exception("Unknown Initial Seed Type")

    #----------
    #Collect result (based on return)
    if solver.stats()["success"] == True:
        print("Round ", roundNum, solver.stats()["success"])
        #Save result/parameters/input(environment) for the current optimization round
        SingleOptResult = {"var_idx":var_index,
                           "opt_res":x_opt,
                           "LeftSwingFlag":InitConfig["LeftSwingFlag"],   "RightSwingFlag":InitConfig["RightSwingFlag"],
                           "x_init": InitConfig["x_init"],   "y_init": InitConfig["y_init"], "z_init": InitConfig["z_init"],
                           "xdot_init": InitConfig["xdot_init"], "ydot_init": InitConfig["ydot_init"], "zdot_init": InitConfig["zdot_init"],
                           "Lx_init": InitConfig["Lx_init"], "Ly_init": InitConfig["Ly_init"], "Lz_init": InitConfig["Lz_init"],
                           "Ldotx_init": InitConfig["Ldotx_init"], "Ldoty_init": InitConfig["Ldoty_init"], "Ldotz_init": InitConfig["Ldotz_init"],
                           "LeftInitSurf":InitConfig["LeftInitSurf"],     "RightInitSurf":InitConfig["RightInitSurf"], #Vertice Representation
                           "PLx_init":InitConfig["PLx_init"],   "PLy_init":InitConfig["PLy_init"],   "PLz_init":InitConfig["PLz_init"],
                           "PRx_init":InitConfig["PRx_init"],   "PRy_init":InitConfig["PRy_init"],   "PRz_init":InitConfig["PRz_init"],
                           "PL_init_TangentX":InitConfig["PL_init_TangentX"], "PL_init_TangentY":InitConfig["PL_init_TangentY"], "PL_init_Norm":InitConfig["PL_init_Norm"],
                           "PR_init_TangentX":InitConfig["PR_init_TangentX"], "PR_init_TangentY":InitConfig["PR_init_TangentY"], "PR_init_Norm":InitConfig["PR_init_Norm"],
                           "LeftInitSurfOrientation":InitConfig["LeftInitSurfOrientation"], "RightInitSurfOrientation":InitConfig["RightInitSurfOrientation"],
                           "ContactSurfs":InitConfig["ContactSurfs"], #Vertice representation
                           "SurfTangentsX":InitConfig["SurfTangentsX"],  "SurfTangentsY":InitConfig["SurfTangentsY"],  "SurfNorms":InitConfig["SurfNorms"],
                           "SurfOrientations":InitConfig["SurfOrientations"],
                           "InitConfig":copy.deepcopy(InitConfig),
                           "SolverStats":solver.stats()}

        #print("Single-opt Result x_init (after collecting): ",SingleOptResult["InitConfig"]["x_init"])
        
        #Append to the Container for all the results
        AllOptResult.append(SingleOptResult)
        AllTraj.append(x_opt)
        CasadiParas.append(ParaList)
        print("Total Program Time: ", round(solver.stats()["t_proc_total"],4))
        print("Computation time from Python: ",time_diff)
        
        #Draw Optimization Results
        if showResult == True:
            # Show Optimization Result
            viz.DisplayResults(TerrainModel = TerrainInfo, SingleOptResult = SingleOptResult, AllOptResult = None)
            # Show local obj tracking Result if we enable local obj tracking
            viz.viewLocalObj(InitConfig = InitConfig, LocalObj = LocalObj, CurrentOptResult = SingleOptResult, \
                              groundTruthTrajPath = LocalObjSettings["GroundTruthTraj"], \
                              roundNum = roundNum, TerrainModel = TerrainInfo) if LocalObjSettings["local_obj_tracking_type"] != None else None
        if printResult == True:
            viz.PrintSingleOptTraj(SingleOptResult = SingleOptResult, LocalObjSettings = LocalObjSettings, LocalObj = LocalObj)
    elif solver.stats()["success"] == False:
        print("Fail at Round", roundNum)
        print("Total Program Time: ", round(solver.stats()["t_proc_total"],4))
        print("Computation time from Python: ",time_diff)

        #Save Failure info, basically just save initial conditions
        FailedRoundInfo = {"var_idx":var_index,
                           "opt_res":x_opt,
                           "Failed_RoundIdx": roundNum,
                           "LeftSwingFlag":InitConfig["LeftSwingFlag"],   "RightSwingFlag":InitConfig["RightSwingFlag"],
                           "x_init": InitConfig["x_init"],   "y_init": InitConfig["y_init"], "z_init": InitConfig["z_init"],
                           "xdot_init": InitConfig["xdot_init"], "ydot_init": InitConfig["ydot_init"], "zdot_init": InitConfig["zdot_init"],
                           "Lx_init": InitConfig["Lx_init"], "Ly_init": InitConfig["Ly_init"], "Lz_init": InitConfig["Lz_init"],
                           "Ldotx_init": InitConfig["Ldotx_init"], "Ldoty_init": InitConfig["Ldoty_init"], "Ldotz_init": InitConfig["Ldotz_init"],
                           "LeftInitSurf":InitConfig["LeftInitSurf"],     "RightInitSurf":InitConfig["RightInitSurf"], #Vertice Representation
                           "PLx_init":InitConfig["PLx_init"],   "PLy_init":InitConfig["PLy_init"],   "PLz_init":InitConfig["PLz_init"],
                           "PRx_init":InitConfig["PRx_init"],   "PRy_init":InitConfig["PRy_init"],   "PRz_init":InitConfig["PRz_init"],
                           "PL_init_TangentX":InitConfig["PL_init_TangentX"], "PL_init_TangentY":InitConfig["PL_init_TangentY"], "PL_init_Norm":InitConfig["PL_init_Norm"],
                           "PR_init_TangentX":InitConfig["PR_init_TangentX"], "PR_init_TangentY":InitConfig["PR_init_TangentY"], "PR_init_Norm":InitConfig["PR_init_Norm"],
                           "LeftInitSurfOrientation":InitConfig["LeftInitSurfOrientation"], "RightInitSurfOrientation":InitConfig["RightInitSurfOrientation"],
                           "ContactSurfs":InitConfig["ContactSurfs"], #Vertice representation
                           "SurfTangentsX":InitConfig["SurfTangentsX"],  "SurfTangentsY":InitConfig["SurfTangentsY"],  "SurfNorms":InitConfig["SurfNorms"],
                           "SurfOrientations":InitConfig["SurfOrientations"],
                           "InitConfig":copy.deepcopy(InitConfig),
                           "SolverStats":solver.stats()}

        break
    #-----------------

#-----------------
#   Show the Result for all the connected Execution Horizon
if showResult == True:
    viz.DisplayResults(TerrainModel = TerrainInfo, SingleOptResult = None, AllOptResult = AllOptResult)

#--------------
#NOTE: Compute Accumulated Cost?

#--------------
#Dump Data into Pickled File
DumpedResults = {"ExternalParameters": ExternalParameters, #External Parameters Passed into Computation
                 "NumLookAhead": NumLookAhead,  "Num_of_Rounds": Nrounds,  "NumLocalKnots": N_knots_per_phase,  "PhaseDurationLimits": phase_duration_limits,
                 "TerrainModel": TerrainInfo["AllPatchesVertices"], #all patches for TSID code
                 "TerrainInfo":  TerrainInfo, #full terrain info (vertices, initial patches, etc)
                 "TerrainSettings":TerrainSettings, #parameter for Terrain Generation Functions
                 "TerrainModelPath":TerrainModelPath,#If get Terrain Model from file, what is the path
                 "LocalObjSettings":LocalObjSettings,
                 "VarIdx_of_All_Levels": var_index,
                 "Trajectory_of_All_Rounds":AllTraj, #all collected x_opt (ptimization result vector)
                 "SingleOptResultSavings":AllOptResult,
                 "SwingLeftFirst":SwingLeftFirst,    #indicator of which foot will swing first for the first step
                 "SwingRightFirst":SwingRightFirst,  #indicator of which foot will swing first for the first step
                 "CasadiParameters":CasadiParas,
                 }

#Save Info For the Failed Round
DumpedResults["FailedRoundInfo"] = FailedRoundInfo if not (len(AllOptResult) == Nrounds) else None 

#print("Single-opt Result x_init (after dumping): ",DumpedResults["SingleOptResultSavings"][0]["InitConfig"]["x_init"])

if saveData == True:
    pickle.dump(DumpedResults, open("/home/jiayu/Desktop/MLP_DataSet/GroundTruthTraj/uneven_plan.p", "wb"))    #Save Data
    sys.stdout.close();   sys.stdout=stdoutOrigin                       #Close logging