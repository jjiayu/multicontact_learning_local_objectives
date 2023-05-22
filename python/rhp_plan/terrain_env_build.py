#Scripts Building the environment model for optimization as well as the gazebo world file for simulation

from ast import Constant
import os

from multicontact_learning_local_objectives.python.terrain_create import *
import multicontact_learning_local_objectives.python.visualization as viz
from multicontact_learning_local_objectives.python.terrain_create.geometry_utils import *
import pickle
import sys

# Check if we have the world file genrated or not
world_file_path = "/home/jiayu/catkin_ws/src/pal_gazebo_worlds_slmc/worlds/uneven_terrain.world"
terrain_model_file_path = "/home/jiayu/Desktop/MLP_DataSet/Env_Models/uneven_terrain.p"
terrain_model_logfile_path = "/home/jiayu/Desktop/MLP_DataSet/Env_Models/uneven_terrain_generate_log.txt"

# Stop the program if we have uneven terrain world file already existed
if os.path.isfile(world_file_path):
    raise Exception("Uneven Terrain World Description File Already Exists")

#Logging command line output
logging = True
if logging == True: 
    stdoutOrigin=sys.stdout; sys.stdout = open(terrain_model_logfile_path, "w")

#-------------------------------------
#For terrain generation
#Make Terrain Setting

doormat_height = 0.041 #0.05
front_bar_switch = True#True

patch_height = 0.025

TerrainSettings = {"terrain_type": "customized",#make sure we set customized terrain
                   "twosteps_on_patch": False,
                   "backward_motion": False,
                   #---------
                   #forward motions
                   #"customized_terrain_pattern": ["X_positive",  "X_negative",    "X_positive",   "X_negative",   "X_positive",   "X_negative",   "X_positive",   "X_negative"], #v-shape
                   #"customized_terrain_pattern": ["Y_negative",  "Y_negative",    "Y_positive",   "Y_positive",   "Y_negative",   "Y_negative",   "Y_positive",   "Y_positive"], #up and down
                   #"customized_terrain_pattern": ["Y_negative",  "X_negative",    "Y_positive",   "X_positive",   "X_negative",   "Y_negative",   "Y_positive",   "Y_positive", "Y_negative",   "Y_negative"], #random
                   #"customized_terrain_pattern": ["DiagX_positive",  "DiagX_negative",    "DiagY_positive",   "DiagY_negative", "DiagX_positive",  "DiagX_negative",    "DiagY_positive",   "DiagY_negative"], #diag example
                   #"customized_terrain_pattern": ["Y_negative",  "DiagX_negative",    "Y_positive",   "X_positive",   "X_negative",   "Y_negative",   "Y_positive",   "Y_positive", "Y_negative",   "Y_negative"], #mixed
                   #"customized_terrain_pattern": ["X_positive",  "X_negative",    "Y_positive",   "Y_negative"], #straight example
                   # almost flat
                   #"customized_terrain_pattern": ["Y_negative",  "DiagX_negative",    "Y_positive",   "DiagY_positive",   "DiagY_negative",   "Y_negative",   "Y_positive",   "Y_positive", "Y_negative",   "Y_negative"], #mixed
                   
                   #Very cutomized
                   #"customized_terrain_pattern": ["Y_negative",  "Y_negative",    "DiagX_positive",   "Y_positive",   "Y_negative",   "DiagX_negative",   "Y_positive",   "DiagY_positive"], #mixed
                   
                   #V-shape innner
                   ##"customized_terrain_pattern": ["X_positive",  "X_negative",    "X_positive",   "X_negative",   "X_positive",   "X_negative"], #mixed
                   # "customized_terrain_pattern": ["X_positive",  "X_negative",    "X_positive",   "X_negative"], #mixed

                   #V-shape outter (Robot Slips)
                   #"customized_terrain_pattern": ["X_negative",  "X_positive",    "X_negative",   "X_positive"], #mixed
                   
                   #Up and down hill
                   #"customized_terrain_pattern": ["Y_negative",  "Y_negative",    "Y_positive",   "Y_positive"], #mixed

                #    #v-shape
                #    "customized_terrain_pattern": ["X_positive",  "X_negative",    
                #                                   "X_positive",  "X_negative", 
                #                                   "X_positive",  "X_negative", 
                #                                   "X_positive",  "X_negative", 
                #                                   "X_positive",  "X_negative",
                #                                   "flat",        "flat"], #mixed


                #    #up-down hills
                #    "customized_terrain_pattern": ["Y_negative",  "Y_negative",    
                #                                   "Y_positive",  "Y_positive", 
                #                                   "Y_negative",  "Y_negative",
                #                                   "Y_positive",  "Y_positive", 
                #                                    "flat",        "flat",
                #                                    "flat",        "flat",], #mixed
                #    #stairs
                #    "customized_terrain_pattern": ["flat",  "flat",    
                #                                   "flat",  "flat",
                #                                   "flat",  "flat",
                #                                   "flat",  "flat", 

                #                                   "flat",  "flat",
                #                                   "flat",  "flat",], #mixed

                   #uneven terrain

                   "customized_terrain_pattern": ["DiagX_positive",  "DiagY_positive",
                                                  "DiagY_negative",  "DiagX_negative", 
                                                  "Y_positive",  "Y_positive",
                                                  "Y_negative",  "Y_negative",
                                                  "DiagX_positive",  "DiagY_positive",
                                                  "DiagY_negative",  "DiagX_negative", 
                                                  "Y_positive",  "Y_positive",
                                                  "Y_negative",  "Y_negative",
                                                  "flat",  "flat",
                                                  "flat",  "flat",], #mixed

                #    #uneven terrain changing
                #    "customized_terrain_pattern": ["Y_negative",  "Y_negative",
                #                                   "Y_positive",  "Y_positive", 
                #                                   "flat",  "flat",
                #                                   "flat",  "flat",
                #                                   "Y_negative",  "Y_negative",
                #                                   "Y_positive",  "Y_positive", 
                #                                   "flat",  "flat",
                #                                   "flat",  "flat",], #mixed

                   #---------
                   #backword motions
                   #almost flat
                   #"customized_terrain_pattern": ["Y_positive",  "DiagY_positive",    "Y_negative",   "DiagX_negative",   "DiagX_positive",   "Y_positive",   "Y_negative",   "Y_negative", "Y_positive",   "Y_positive"], #mixed
                   #"fixed_inclination":          [10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi],
                   #"fixed_inclination":          [10.0/180*np.pi, 10.0/180*np.pi, 15.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 15.0/180*np.pi, 10.0/180*np.pi],#None,#0.0/180*np.pi, #radius, None means random inclination
                   
                   #----------
                #    #Terrain height vector
                   "customized_terrain_height_flag": True,
                   "customized_terrain_height_list": [0.0, 0.0, 
                                                      0.0, 0.0,
                                                      0.0, 0.0,
                                                      0.0, 0.0,
                                                      0.0, 0.0,
                                                      0.0, 0.0,
                                                      0.0, 0.0,
                                                      0.0, 0.0,
                                                      0.0, 0.0,
                                                      0.0, 0.0,],
                   #Stairs
                #    "customized_terrain_height_flag": True,
                #    "customized_terrain_height_list": [0.0, 0.0, 
                #                                       0.0, 0.0,
                #                                       0.041, 0.041, 
                #                                       0.041, 0.041],
                   
                   "fixed_inclination":10/180*np.pi,
                   "lab_blocks": True, #make sure this is true to have the same patches as the lab env
                   "lab_block_z_shift": 0.006 + 0.018, #measured do not change, bottom + surface height
                   "inner_blocks": False, "inner_block_length": 0.27,
                   "gap_index": [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],#[8,9,10,11], #the patch that has a gap
                   "gap_dist": 0.015, # the gap dist
                   "random_init_surf_size": False,
                   "random_surfsize_flag": False,
                   "random_Horizontal_Move": False,
                   "MisMatch_Alignment_of_FirstTwoPatches": True, #bool(np.random.choice([True,False],1)), 
                   "MisAligned_Column": "right", #can be "left", "right", None (choose randomly)
                   "MisAligned_Amount": 0.0, #0.25
                   "Gap_Between_Patches": True,
                   "Gap_along_x": 0.0, #0.3
                   "Projected_Length": 0.3, "Projected_Width": 0.3, #0.55 and 1.0
                   "large_slope_flag":False,
                   "large_slope_index": [],#[np.random.choice([16,17])],#select a patch from number 16 or 17
                   "large_slope_directions": [],#[np.random.choice(["X_positive", "X_negative", "Y_positive", "Y_negative"])], 
                   "large_slope_inclinations": [],#[23/180*np.pi],#[np.round(np.random.uniform(17.0/180*np.pi,25.0/180*np.pi),3)], #if no elevation change, 22 degress is the limit
                   "large_slope_X_shifts": [],#[0.0], 
                   "large_slope_Y_shifts": [],#[0.0], 
                   "large_slope_Z_shifts": [],#[np.random.uniform(-0.25,0.25)],
                   "y_center_offset": 0.0,
                   "x_offset": 0.0,
                   "constant_block_z_shift_flag": True, #True for doormat case
                   "constant_block_z_shift_value": -doormat_height,
                    }

if TerrainSettings["terrain_type"] == "customized":
    TerrainSettings["num_of_steps"] = len(TerrainSettings["customized_terrain_pattern"])
else:
    TerrainSettings["num_of_steps"] = 30

# Generate terrain model (for python)
terrain_model = terrain_model_gen_lab_inner_blocks(terrain_name    = TerrainSettings["terrain_type"],  
                                      customized_terrain_pattern = TerrainSettings["customized_terrain_pattern"],
                                      customized_terrain_height_flag = TerrainSettings["customized_terrain_height_flag"],
                                      customized_terrain_height_list = TerrainSettings["customized_terrain_height_list"],
                                      fixed_inclination = TerrainSettings["fixed_inclination"], 
                                      gap_index = TerrainSettings["gap_index"], #gap in particular patches
                                      gap_dist = TerrainSettings["gap_dist"],#gap in particular patches
                                      backward_motion=TerrainSettings["backward_motion"],
                                      lab_blocks = TerrainSettings["lab_blocks"],
                                      lab_block_z_shift = TerrainSettings["lab_block_z_shift"],
                                      inner_block_length=TerrainSettings["inner_block_length"],
                                      inner_blocks=TerrainSettings["inner_blocks"],
                                      randomInitSurfSize = TerrainSettings["random_init_surf_size"], #False,
                                      random_surfsize = TerrainSettings["random_surfsize_flag"],
                                      randomHorizontalMove = TerrainSettings["random_Horizontal_Move"],
                                      randomMisAlignmentofFirstTwoPatches = TerrainSettings["MisMatch_Alignment_of_FirstTwoPatches"], 
                                      MisAlignmentColumn = TerrainSettings["MisAligned_Column"], 
                                      MisAlignmentAmount = TerrainSettings["MisAligned_Amount"],
                                      gap_between_patches = TerrainSettings["Gap_Between_Patches"],
                                      x_gap = TerrainSettings["Gap_along_x"],
                                      Proj_Length = TerrainSettings["Projected_Length"], Proj_Width = TerrainSettings["Projected_Width"],
                                      NumSteps = TerrainSettings["num_of_steps"], 
                                      NumLookAhead = 15,#Put NumLookAhead = 20 to give infinitely long terrains
                                      large_slope_flag = TerrainSettings["large_slope_flag"], 
                                      large_slope_index = TerrainSettings["large_slope_index"], large_slope_directions = TerrainSettings["large_slope_directions"], 
                                      large_slope_inclinations = TerrainSettings["large_slope_inclinations"],
                                      large_slope_X_shifts = TerrainSettings["large_slope_X_shifts"], 
                                      large_slope_Y_shifts = TerrainSettings["large_slope_Y_shifts"],
                                      large_slope_Z_shifts = TerrainSettings["large_slope_Z_shifts"],
                                      y_center = TerrainSettings["y_center_offset"],
                                      x_offset = TerrainSettings["x_offset"],
                                      twosteps_on_patch = TerrainSettings["twosteps_on_patch"],
                                      Constant_Block_Z_Shift_Flag = TerrainSettings["constant_block_z_shift_flag"],
                                      Constant_Block_Z_Shift_Value = TerrainSettings["constant_block_z_shift_value"]) 

viz.DisplayResults(TerrainModel=terrain_model, SingleOptResult=None, AllOptResult=None)

# save the terrain model
env_model_to_save = {"TerrainModel": terrain_model["AllPatchesVertices"], # only the vertices
                     "TerrainInfo":  terrain_model, #all information for the terrain model
                     "TerrainSettings": TerrainSettings, #setting of the terrain
}

pickle.dump(env_model_to_save, open(terrain_model_file_path, "wb"))

#-----------------------------
# Make the world file

# Check if we have the correct 
if (TerrainSettings["Projected_Length"] == 0.4 and TerrainSettings["Projected_Width"] == 0.4) or (TerrainSettings["Projected_Length"] == 0.3 and TerrainSettings["Projected_Width"] == 0.3):
    print("We have the terrain models")
else:
    raise Exception("We don't have the terrain models")

# Make a world file (with ground floor) if we dont have one
with open(world_file_path, 'x') as f:
    f.write('<?xml version="1.0" ?>\n')
    f.write('<sdf version="1.4">\n')
    f.write('  <world name="default">\n')
    f.write('    <physics type="ode">\n')
    f.write('      <gravity>0 0 -9.81</gravity>\n')
    f.write('      <ode>\n')
    f.write('        <solver>\n')
    f.write('          <type>quick</type>\n')
    f.write('          <iters>50</iters>\n')
    f.write('          <sor>1.4</sor>\n')
    f.write('        </solver>\n')
    f.write('        <constraints>\n')
    f.write('          <cfm>0.0</cfm>\n')
    f.write('          <erp>0.2</erp>\n')
    f.write('          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>\n')
    f.write('          <contact_surface_layer>0.0</contact_surface_layer>\n')
    f.write('        </constraints>\n')
    f.write('      </ode>\n')
    f.write('      <real_time_update_rate>1000</real_time_update_rate>\n')
    f.write('      <max_step_size>0.001</max_step_size>\n')
    f.write('    </physics>\n')
    f.write('\n')
    f.write('    <!-- A global light source -->\n')
    f.write('    <include>\n')
    f.write('      <uri>model://sun</uri>\n')
    f.write('    </include>\n')
    f.write('\n')
    f.write('    <!-- A ground plane -->\n')
    f.write('    <include>\n')
    f.write('      <uri>model://ground_plane</uri>\n')
    f.write('    </include>\n')
    f.write('\n')

    if front_bar_switch == True:
        # Write the front bar
        f.write('    <!-- Place the front bar -->\n')
        f.write('    <include>\n')
        f.write('      <name>font_bar</name>\n')
        f.write('      <static>'+ str(1) +'</static>\n')
        init_surf_border_x = terrain_model["AllPatchesVertices"][0][0][0]-0.01
        init_surf_border_y = terrain_model["AllPatchesVertices"][0][2][1]
        f.write('      <uri>model://front_bar</uri>\n')
        f.write('      <pose> ' + str(init_surf_border_x) + " " + str(init_surf_border_y) + " " + str(-doormat_height) + ' 0.0 0.0 ' + str(0.0) + '</pose>\n')
        f.write('    </include>\n')
        f.write('\n')

# # construct the gazebo world file, append terrain models
with open(world_file_path, 'a') as f:
    for surf_idx in range(2,len(terrain_model["AllPatchesVertices"])):
        #temp_type = terrain_model["AllPatchesVertices"][surf_idx]
        temp_surf = terrain_model["AllPatchesVertices"][surf_idx]
        temp_center_x, temp_center_y, temp_center_z = getCenter(temp_surf)
        #get min z of the patch
        cur_patch_min_z = np.min([temp_surf[0,2], temp_surf[1,2], temp_surf[2,2], temp_surf[3,2]])
        temp_type = getSurfaceType(temp_surf)
        temp_surf_inclination = abs(getTerrainRotationAngle(temp_surf))#terrain_model["ContactSurfsInclinationsDegrees"][surf_idx]
        if temp_type != "flat": #if NOT flat add the terrain blocks
            if temp_type == "X_positive":
                yaw_rot_angle = np.pi/2*3 #rotate 3/4 circle
            elif temp_type == "X_negative":
                yaw_rot_angle = np.pi/2*1.0 #rotate a quater of a circle
            elif temp_type == "Y_positive":
                yaw_rot_angle = np.pi/2*0.0 #no rotation
            elif temp_type == "Y_negative":
                yaw_rot_angle = np.pi/2*2 # rotate half of the circle
            elif temp_type == "DiagX_positive":
                yaw_rot_angle = np.pi/2*3.0 #rotate 3/4 circle
            elif temp_type == "DiagX_negative":
                yaw_rot_angle = np.pi/2*1.0 #rotate a quater of a circle
            elif temp_type == "DiagY_positive":
                yaw_rot_angle = np.pi/2*0.0 #no rotation
            elif temp_type == "DiagY_negative":
                yaw_rot_angle = np.pi/2*2 # rotate half of the circle
            else:
               raise Exception("Unrecognised Terrain type")

            print(temp_type)
            
            f.write('    <!-- Place a Block -->\n')
            f.write('    <include>\n')
            f.write('      <name>block_'+str(surf_idx)+'</name>\n')
            f.write('      <static>'+ str(1) +'</static>\n')
            if temp_type in ["X_positive","X_negative","Y_positive","Y_negative"]:
                if TerrainSettings["Projected_Length"] == 0.4:
                    f.write('      <uri>model://Y_positive_' + str(int(temp_surf_inclination)) + '_lifted</uri>\n')
                elif TerrainSettings["Projected_Length"] == 0.3:
                    f.write('      <uri>model://Y_positive_' + str(int(temp_surf_inclination)) + '_lifted_size_30</uri>\n')
            elif temp_type in ["DiagX_positive", "DiagX_negative", "DiagY_positive", "DiagY_negative"]:
                if TerrainSettings["Projected_Length"] == 0.4:
                    f.write('      <uri>model://diag_Y_positive_' + str(int(temp_surf_inclination)) + '_lifted</uri>\n')
                elif TerrainSettings["Projected_Length"] == 0.3:
                    f.write('      <uri>model://diag_Y_positive_' + str(int(temp_surf_inclination)) + '_lifted_size_30</uri>\n')
            else:
                raise Exception("Unknown terrain type")
            f.write('      <pose> ' + str(temp_center_x) + " " + str(temp_center_y) + " " + str(cur_patch_min_z - TerrainSettings["lab_block_z_shift"]) + ' 0.0 0.0 ' + str(yaw_rot_angle) + '</pose>\n')
            f.write('    </include>\n')
            f.write('\n')
        
        if temp_type == "flat": #if the terrain type is flat
            terrain_height_temp = temp_surf[0][2]

            if terrain_height_temp > 0: #only add block when we have height larger than 0
                f.write('    <!-- Place a (flat) Block -->\n')
                f.write('      <model name='+'\"block_' + str(surf_idx) + '\">\n')
                #get terrain height
                f.write('      <pose> ' + str(temp_center_x) + " " + str(temp_center_y) + " " + str(0 + terrain_height_temp/2.0) + ' 0.0 0.0 0.0' + '</pose>\n')
                f.write('      <static>'+ str(1) +'</static>\n')
                f.write('      <link name='+ '\"block_' + str(surf_idx) + '\">\n')
                f.write('      <inertial>\n')
                f.write('      <mass>1.0</mass>\n')
                f.write('      <inertia>\n')
                f.write('        <ixx>1.0</ixx>\n')
                f.write('        <ixy>0.0</ixy>\n')
                f.write('        <ixz>0.0</ixz>\n')
                f.write('        <iyy>1.0</iyy>\n')
                f.write('        <iyz>0.0</iyz>\n')
                f.write('        <izz>1.0</izz>\n')
                f.write('        </inertia>\n')
                f.write('        </inertial>\n')
                f.write('        <collision name=\"collision\">\n')
                f.write('        <geometry>\n')
                f.write('          <box>\n')
                f.write('          <size>' + str(TerrainSettings["Projected_Length"]) + ' ' + str(TerrainSettings["Projected_Width"]) + ' ' + str(terrain_height_temp) + '</size>\n')
                f.write('          </box>\n')
                f.write('        </geometry>\n')
                f.write('        </collision>\n')
                f.write('        <visual name=\"visual\">\n')
                f.write('        <geometry>\n')
                f.write('          <box>\n')
                f.write('          <size>' + str(TerrainSettings["Projected_Length"]) + ' ' + str(TerrainSettings["Projected_Width"]) + ' ' + str(terrain_height_temp) + '</size>\n')
                f.write('          </box>\n')
                f.write('        </geometry>\n')
                f.write('        </visual>\n')
                f.write('      </link>\n')
                f.write('      </model>\n')

#Closing the world file
with open(world_file_path, 'a') as f:
    f.write('  </world>\n')
    f.write('</sdf>\n')

if logging == True:
    sys.stdout.close();   sys.stdout=stdoutOrigin  
