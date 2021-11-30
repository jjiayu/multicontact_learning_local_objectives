# multicontact_learning_local_objectives
Learning Local Objective for Guiding Multi-Contact Locomotion


## Python Package Arrangement
- Currently use global package management: in __init__.py write "from .module_name/filename import *" (note the dot there to identify we are getting module from current folder), then call the function from import package_name as xxx, package_name.function)name()
- For large packages:
- Each subfolder in src represent a python package (with an __init__.py).
- In each __init__.py, we add line: import package_name(subfolder_name).module_name(py_filename) 
- When use the function/modules, either use: from package_name import module_name or import packagename.module_name as xxx, then use module_name.function() or xxx.function(); or we can use import package_name direclty and then package_name.module_name.function_name()

## Load functions/packages from parent folder or parent's folder subfolder
- No good ways, write absolute path, i.e. import multicontact_learning.src.package_name as package_alias

## Notes
- AllTerrain is a list of all patches, [pending] the first element is init left patch, the second element is the init right patch, all the rest are the patches for all following steps

- Gait Pattern and Step Numbers
Whole_Lookahead:     Step_0     Step_1     Step_2     Step_3     Step_4....
Second Level Step:             (Step_0)   (Step_1)   (Step_2)   (Step_3)....
Swing Left First:      L          R          L          R          L
Swing Right First:     R          L          R          L          R

## Talos Joint Assignment
Joint 1:  hip   - yaw   - z-axis
Joint 2:  hip   - roll  - x-axis
Joint 3:  hip   - pitch - y-axis
Joint 4:  knee  - pitch - y-axis
Joint 5:  ankle - pitch - y-axis
Joint 6:  ankle - roll  - x-axis

NOTE: hip roll pitch limits need to be consistent with ankle joint, otherwise the robot cannot standstill
____________
\
 \
  \ 
   \
    \
 ____\______   


## Compute DataPoints
longjob -28day -c "nice bash server_datapoint_compute.sh Flat_and_OneLargeSlope RawTrainingSetRollOuts Group1 50"