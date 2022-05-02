#Scripts Building the environment model for optimization as well as the gazebo world file for simulation

import os
from multicontact_learning_local_objectives.python.terrain_create import *
import multicontact_learning_local_objectives.python.visualization as viz

# Check if we have the world file genrated or not
world_file_path = "/home/jiayu/catkin_ws/src/pal_gazebo_worlds_slmc/worlds/uneven_terrain.world"

# Stop the program if we have uneven terrain world file already existed
if os.path.isfile(world_file_path):
    raise Exception("Uneven Terrain World Description File Already Exists")

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
    f.write('    <!-- A global light source -->\n')
    f.write('    <include>\n')
    f.write('      <uri>model://sun</uri>\n')
    f.write('    </include>\n')
    f.write('    <!-- A ground plane -->\n')
    f.write('    <include>\n')
    f.write('      <uri>model://ground_plane</uri>\n')
    f.write('    </include>\n')

# Generate terrain model (for python)
env_model = terrain_model_gen_lab(terrain_name="customized",
                      customized_terrain_pattern = ["X_positive"]*10,
                      Proj_Length=0.6, Proj_Width=0.6,
                      fixed_inclination=10.0/np.pi,
                      randomInitSurfSize=False,
                      random_surfsize=False, min_shrink_factor=0.0, max_shrink_factor=0.3,
                      randomHorizontalMove=False,  # Need to add random Height Move for Normal Patches
                      randomElevationShift=False, min_elevation_shift=-0.075, max_elevation_shift=0.075,
                      randomMisAlignmentofFirstTwoPatches=False, MisAlignmentColumn=None, MisAlignmentAmount=0.25,
                      NumSteps=10, NumLookAhead=4,
                      large_slope_flag=False, large_slope_index=[8],
                      large_slope_directions=["X_positive"], large_slope_inclinations=[18.0/180.0*np.pi],
                      large_slope_X_shifts=[0.0], large_slope_Y_shifts=[0.0], large_slope_Z_shifts=[0.0],
                      y_center = 0.0,
                      x_offset = 0.0)

viz.DisplayResults(TerrainModel=env_model, SingleOptResult=None, AllOptResult=None)

# save the terrain model

# # construct the gazebo world file, append terrain models
# with open(world_file_path, 'a') as f:
#     f.write('    <!-- A bunch of chairs -->\n')
#     f.write('    <include>\n')
#     f.write('          <uri>model://Y_positive_10_lifted</uri>\n')
#     f.write('      <pose>1.0 0 0 0 0 0.0</pose>\n')
#     f.write('    </include>\n')


#Closing the world file
with open(world_file_path, 'a') as f:
    f.write('  </world>\n')
    f.write('</sdf>\n')
