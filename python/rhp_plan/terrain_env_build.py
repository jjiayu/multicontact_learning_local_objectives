#Scripts Building the environment model for optimization as well as the gazebo world file for simulation

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
                   "customized_terrain_pattern": ["Y_negative",  "Y_negative",    "DiagX_positive",   "Y_positive",   "Y_negative",   "DiagX_negative",   "Y_positive",   "DiagY_positive"], #mixed
                   #---------
                   #backword motions
                   #almost flat
                   #"customized_terrain_pattern": ["Y_positive",  "DiagY_positive",    "Y_negative",   "DiagX_negative",   "DiagX_positive",   "Y_positive",   "Y_negative",   "Y_negative", "Y_positive",   "Y_positive"], #mixed
                   #"fixed_inclination":          [10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi],
                   #"fixed_inclination":          [10.0/180*np.pi, 10.0/180*np.pi, 15.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 10.0/180*np.pi, 15.0/180*np.pi, 10.0/180*np.pi],#None,#0.0/180*np.pi, #radius, None means random inclination
                   "fixed_inclination":10/180*np.pi,
                   "lab_blocks": True, #make sure this is true to have the same patches as the lab env
                   "lab_block_z_shift": 0.006 + 0.018, #measured do not change, bottom + surface height
                   "inner_blocks": False, "inner_block_length": 0.27,
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
                   "y_center_offset": 0.0,
                   "x_offset": 0.0,
                    }

if TerrainSettings["terrain_type"] == "customized":
    TerrainSettings["num_of_steps"] = len(TerrainSettings["customized_terrain_pattern"])
else:
    TerrainSettings["num_of_steps"] = 30

# Generate terrain model (for python)
terrain_model = terrain_model_gen_lab_inner_blocks(terrain_name    = TerrainSettings["terrain_type"],  
                                      customized_terrain_pattern = TerrainSettings["customized_terrain_pattern"],
                                      fixed_inclination = TerrainSettings["fixed_inclination"], 
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
                                      Proj_Length = TerrainSettings["Projected_Length"], Proj_Width = TerrainSettings["Projected_Width"],
                                      NumSteps = TerrainSettings["num_of_steps"], 
                                      NumLookAhead = 10,#Put NumLookAhead = 20 to give infinitely long terrains
                                      large_slope_flag = TerrainSettings["large_slope_flag"], 
                                      large_slope_index = TerrainSettings["large_slope_index"], large_slope_directions = TerrainSettings["large_slope_directions"], 
                                      large_slope_inclinations = TerrainSettings["large_slope_inclinations"],
                                      large_slope_X_shifts = TerrainSettings["large_slope_X_shifts"], 
                                      large_slope_Y_shifts = TerrainSettings["large_slope_Y_shifts"],
                                      large_slope_Z_shifts = TerrainSettings["large_slope_Z_shifts"],
                                      y_center = TerrainSettings["y_center_offset"],
                                      x_offset = TerrainSettings["x_offset"],
                                      twosteps_on_patch = TerrainSettings["twosteps_on_patch"]) 

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

# # construct the gazebo world file, append terrain models
with open(world_file_path, 'a') as f:
    for surf_idx in range(2,len(terrain_model["AllPatchesVertices"])):
        #temp_type = terrain_model["AllPatchesVertices"][surf_idx]
        temp_surf = terrain_model["AllPatchesVertices"][surf_idx]
        temp_center_x, temp_center_y, temp_center_z = getCenter(temp_surf)
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
            f.write('      <pose> ' + str(temp_center_x) + " " + str(temp_center_y) + " " + str(0.0) + ' 0.0 0.0 ' + str(yaw_rot_angle) + '</pose>\n')
            f.write('    </include>\n')
            f.write('\n')

#Closing the world file
with open(world_file_path, 'a') as f:
    f.write('  </world>\n')
    f.write('</sdf>\n')

if logging == True:
    sys.stdout.close();   sys.stdout=stdoutOrigin  
