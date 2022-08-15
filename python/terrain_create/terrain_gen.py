# At present, we assume the contact sequence starts from the left foot contact

import numpy as np
from sl1m.constants_and_tools import *
import matplotlib.pyplot as plt  # Matplotlib
from mpl_toolkits.mplot3d import Axes3D
from multicontact_learning_local_objectives.python.terrain_create.geometry_utils import *
import copy

# NOTE: ALL FUNCTIONS ONLY CONSIDER LEFT SWING IN THE FIRST STEP

#The one we will be using for lab experiments
def terrain_model_gen_lab_inner_blocks(terrain_name=None,
                      backward_motion = False,
                      customized_terrain_pattern = [],
                      Proj_Length=0.6, Proj_Width=0.6,
                      fixed_inclination=0.0,
                      lab_blocks = False, lab_block_z_shift = 0.0,
                      twosteps_on_patch = False,
                      inner_blocks = False, inner_block_length = 0.3,
                      randomInitSurfSize=False,
                      random_surfsize=False, min_shrink_factor=0.0, max_shrink_factor=0.3,
                      randomHorizontalMove=False,  # Need to add random Height Move for Normal Patches
                      randomElevationShift=False, min_elevation_shift=-0.075, max_elevation_shift=0.075,
                      randomMisAlignmentofFirstTwoPatches=False, MisAlignmentColumn=None, MisAlignmentAmount=0.25,
                      Constant_Block_Z_Shift_Flag = False,
                      Constant_Block_Z_Shift_Value = 0.0,
                      gap_between_patches = False,
                      x_gap = 0.0,
                      NumSteps=None, NumLookAhead=None,
                      large_slope_flag=False, large_slope_index=[],
                      large_slope_directions=[], large_slope_inclinations=[],
                      large_slope_X_shifts=[], large_slope_Y_shifts=[], large_slope_Z_shifts=[],
                      y_center = 0.0,
                      x_offset = 0.0,
                      ):

    #twosteps_on_patch and inner_blocks cannot be true at the same time
    if twosteps_on_patch == True and inner_blocks == True:
        raise Exception("twosteps_on_patch and inner_blocks cannot be true simultaneousely")

    # Check large slope parameters are in the same length
    if large_slope_flag == True:
        if (len(large_slope_index) == len(large_slope_directions)) and (len(large_slope_directions) == len(large_slope_inclinations)) and \
           (len(large_slope_inclinations) == len(large_slope_X_shifts)) and (len(large_slope_X_shifts) == len(large_slope_Y_shifts)) and \
           (len(large_slope_Y_shifts) == len(large_slope_Z_shifts)):
            print("All Large Slope Parameters in the same Length")
        else:
            raise Exception("Large Slope Parameters in a Different Length")

    # Surf Vertex Identification
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4

    # ---------------
    # Decide Parameters
    # Number of Surfaces based on Number of steps and Number of lookahead and initial patches
    TotalNumSurfs = NumSteps + NumLookAhead - 1 + 2  # (include the intial two contact patches

    # ---------------
    # Decide Which column will have Mis Alignment of (the first two patches) if not provided
    if randomMisAlignmentofFirstTwoPatches == True:
        if MisAlignmentColumn == None:
            MisAlignmentColumn = np.random.choice(["left", "right"], 1)
    elif randomMisAlignmentofFirstTwoPatches == False:
        MisAlignmentColumn = "left"

    # Convert to the index of the mismatch (NOTE: idx is for COntact Patch sequence, not all patches; start from zero)
    if MisAlignmentColumn == "left":
        MisAlignedPatchIdx = 0
    elif MisAlignmentColumn == "right":
        MisAlignedPatchIdx = 1
    else:
        raise Exception("Unknown column name")

    # ---------------
    # Generate Initial Patches (Currently Define as flat patches)
    if randomInitSurfSize == False:
        
        if backward_motion == True: #simulator footstep initialization is negative, put it a bit further for backward motion
            InitContactSurf_x_max = 0.11 + 0.02 + x_offset #0.11 the half foot size 0.02 the front bar
        elif backward_motion == False: #for forward motion is ok
            InitContactSurf_x_max = 0.11 + 0.02 + x_offset #0.16 + x_offset
    elif randomInitSurfSize == True:
        InitContactSurf_x_max = np.random.uniform(0.115, 0.215) + x_offset
    else:
        raise Exception("Unknow Flag for Init Contact Surf Left most Boder")

    print("Left Init Contact Surface:")
    Sl0 = np.array([[InitContactSurf_x_max, 1.0+y_center, 0.], [-0.2, 1.0+y_center, 0.],
                   [-0.2, y_center, 0.], [InitContactSurf_x_max, y_center, 0.0]])
    #Sl0 = np.array([[0.35, 1.0, 0.], [-0.35, 1.0, 0.], [-0.35, 0.0, 0.], [0.35, 0.0, 0.0]])
    LeftSurfType = getSurfaceType(Sl0)  # "flat"
    print("Left Init Contact Surface Type: ", LeftSurfType)
    Sl0_TangentX, Sl0_TangentY, Sl0_Norm, Sl0_Orientation = getTerrainTagentsNormOrientation(
        Sl0)

    print("Right Init Contact Surface")
    Sr0 = np.array([[InitContactSurf_x_max, y_center, 0.], [-0.2, y_center, 0.],
                   [-0.2, -1.0 + y_center, 0.], [InitContactSurf_x_max, -1.0 + y_center, 0.0]])
    #Sr0 = np.array([[0.7, 0, 0.], [0.0, 0, 0.], [0.0, -1.0, 0.], [0.7, -1.0, 0.0]])
    RightSurfType = getSurfaceType(Sr0)  # "flat"
    print("Right Init Contact Surface Type: ", RightSurfType)
    Sr0_TangentX, Sr0_TangentY, Sr0_Norm, Sr0_Orientation = getTerrainTagentsNormOrientation(
        Sr0)

    # ------------
    # Make Contact Patches
    # --------------------------
    #   Decide Terrain Pattern
    if (terrain_name == "flat") or (terrain_name == "stair") or (terrain_name == "single_flat"):
        terrain_pattern = ["flat", "flat", "flat", "flat"] * 50  # 200 steps

    elif terrain_name == "antfarm_left":
        # up v first, then down v
        # NOTE: Currently Assume the Initial Contacts are Flat
        terrain_pattern = ["X_negative", "X_positive",
                           "X_positive", "X_negative"] * 50  # 200 steps

    elif terrain_name == "up_and_down_left":
        terrain_pattern = ["Y_negative", "Y_negative",
                           "Y_positive", "Y_positive"] * 50  # 200 steps

    elif terrain_name == "random":
        #TerrainTypeList = ["flat", "X_positive", "X_negative", "Y_positive", "Y_negative"]
        # 4 types of terrain, no flat
        TerrainTypeList = ["X_positive",
                           "X_negative", "Y_positive", "Y_negative"]
        terrain_pattern = np.random.choice(TerrainTypeList, 200)
    elif terrain_name == "random_diag":
        TerrainTypeList = ["X_positive","X_negative", "Y_positive", "Y_negative",
                           "DiagX_positive", "DiagX_negative", "DiagY_positive", "DiagY_negative"]
        terrain_pattern = np.random.choice(TerrainTypeList, 200)
    elif terrain_name == "customized":
        terrain_pattern = customized_terrain_pattern + ["flat"]*200
    else:
        raise Exception("Unknown Terrain Type")

    print(terrain_pattern)

    # -----------------------------------
    #   Create container for Patches
    ContactSurfsVertice = []
    ContactSurfsTypes = []
    ContactSurfsNames = []
    ContactSurfsInclinationsDegrees = []
    AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    # ------------------------------------
    # Generate a sequence of flat patches
    for surfNum in range(2, TotalNumSurfs):
        print("Put Contact Patch ", surfNum)

        # Decide Coordinate of reference point
        if surfNum % 2 == 0:  # Even number of steps (Swing the Left)
            # ----       #p2---------------------p1
            #            |                      |
            # Sn-2(Sl0) # |         Sn           |
            # (L) #      |        (L)           |
            # ----      #p3 (ref_x, ref_y)-------p4
            #            |                      |
            # Sn-1(Sr0) # |                      |
            # (R) #      |                      |

            # Define which Contact foot is for the Surface
            SurfContactLeft = "left"
            # ref_x decided by the P1_x of Sn-2 (-2 in index)
            ref_x = AllPatches[-2][0][0]
            # ref_y decided by the P1_y of Sn-1 (-1 in index)
            ref_y = AllPatches[-1][0][1]
            #center = getCenter(AllPatches[-1])
            # center[2] #ref_z decided by the central z of Sn-1 (relative foot height constraints is enforced with respect to stationary foot)
            ref_z = 0.0

        elif surfNum % 2 == 1:  # Odd number of steps
            # ---- #        ---------------------
            #      |                      |
            #      |         Sn-1         |
            #      |          (L)         |
            # ---- #      ---------------------
            #     P2 (ref_x, ref_y)------P1
            #      |                      |
            # Sn-2(Sr0) # |         Sn           |
            # (R) #      |        (R)           |
            #    P3---------------------P4

            # Define which Contact foot is for the Surface
            SurfContactLeft = "right"
            ref_x = AllPatches[-2][0][0]  # ref_x decided by the P1_x of Sn-2
            # ref_y decided by the P3_y of Sn-1 (ContactSurfsVertice[-1])
            ref_y = AllPatches[-1][2][1]
            #center = getCenter(AllPatches[-1])
            # center[2] #ref_z decided by the central z of Sn-1 (relative foot height constraints is enforced with respect to stationary foot)
            ref_z = 0.0

        # Shift along x-direction for mis alignment of First Two Patches
        # Convert from Contact Patch Sequence index (start from 0) to all patches index (+ 2)
        if surfNum == MisAlignedPatchIdx + 2:
            if randomMisAlignmentofFirstTwoPatches == True:  # if we randomly mis align the first two patches
                # MisAlignmentAmount = np.random.uniform(0.0,MaxMisAllignmentAmount) #random choose a mis aligned amount
                ref_x = ref_x + MisAlignmentAmount  # shift the x-ref of the selected patch

        #Give gaps to the patches
        if gap_between_patches == True:
            if surfNum > 2: #we give gap starting from the 4th patch (right column)
                ref_x = ref_x + x_gap

        # Build an initial flat surface
        surf_temp = flat_patch_gen(PatchColumn=SurfContactLeft, ref_x=ref_x,
                                   ref_y=ref_y, ref_z=ref_z, Proj_Length=Proj_Length, Proj_Width=Proj_Width)

        # Add to current surface to lists
        ContactSurfsVertice.append(surf_temp)
        ContactSurfsNames.append("S"+str(surfNum-2))
        ContactSurfsTypes.append(terrain_pattern[surfNum])
        AllPatches.append(surf_temp)

    #backward motion need to change the x coordinates NOTE: we are just modifying the flat patches, so we dont need to change the y and z coordinates
    if backward_motion == True:#flit the x if we turn on the backward motion
        
        #for Sl0
        temp_Sl0 = copy.deepcopy(Sl0)
        #flit vertices 1 and 2 swap, 3 and 4 swap
        Sl0[0] = temp_Sl0[1]; Sl0[0][0] = -Sl0[0][0]
        Sl0[1] = temp_Sl0[0]; Sl0[1][0] = -Sl0[1][0]
        Sl0[2] = temp_Sl0[3]; Sl0[2][0] = -Sl0[2][0]
        Sl0[3] = temp_Sl0[2]; Sl0[3][0] = -Sl0[3][0]

        #for Sr0
        temp_Sr0 = copy.deepcopy(Sr0)
        #flit vertices 1 and 2 swap, 3 and 4 swap
        Sr0[0] = temp_Sr0[1]; Sr0[0][0] = -Sr0[0][0]
        Sr0[1] = temp_Sr0[0]; Sr0[1][0] = -Sr0[1][0]
        Sr0[2] = temp_Sr0[3]; Sr0[2][0] = -Sr0[2][0]
        Sr0[3] = temp_Sr0[2]; Sr0[3][0] = -Sr0[3][0]

        temp_ContactSurfsVertice = copy.deepcopy(ContactSurfsVertice)
        for surf_idx in range(len(ContactSurfsVertice)):
            ContactSurfsVertice[surf_idx][0] = temp_ContactSurfsVertice[surf_idx][1]; ContactSurfsVertice[surf_idx][0][0] = -ContactSurfsVertice[surf_idx][0][0]
            ContactSurfsVertice[surf_idx][1] = temp_ContactSurfsVertice[surf_idx][0]; ContactSurfsVertice[surf_idx][1][0] = -ContactSurfsVertice[surf_idx][1][0]
            ContactSurfsVertice[surf_idx][2] = temp_ContactSurfsVertice[surf_idx][3]; ContactSurfsVertice[surf_idx][2][0] = -ContactSurfsVertice[surf_idx][2][0]
            ContactSurfsVertice[surf_idx][3] = temp_ContactSurfsVertice[surf_idx][2]; ContactSurfsVertice[surf_idx][3][0] = -ContactSurfsVertice[surf_idx][3][0]

    # Update AllPatches with newly genreated ContactSurfVertices
    AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    # -----------------------------------
    # Modify the patches
    # Clear All Patches (because it finishes its duty)
    AllPatches = []
    # ----------------------------------------
    # (Randomly) Shrink size if we want (only make smaller)
    if random_surfsize == True:
        for ContactSurfNum in range(len(ContactSurfsVertice)):
            shrinkFactor = np.random.uniform(
                min_shrink_factor, max_shrink_factor)
            ContactSurfsVertice[ContactSurfNum] = flat_patch_shrink(
                surf=ContactSurfsVertice[ContactSurfNum], shrinkFactor=shrinkFactor, Proj_Length=Proj_Length, Proj_Width=Proj_Width)
            print("Shrink (Smaller) Contact Factor of Patch ",
                  ContactSurfNum, "with Factor of ", shrinkFactor)

    # Update AllPatches with newly genreated ContactSurfVertices
    AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    # -----------------------------------------------------------
    # Move the patches (in X Y directions only) (NOTE: We need to ensure no over shoot)
    if randomHorizontalMove == True:
        MovingDirectionList = ["X_positive",
                               "X_negative", "Y_positive", "Y_negative"]
        for surfNum in range(2, TotalNumSurfs):
            # Decide Coordinate of reference point
            if surfNum % 2 == 0:  # Even number of steps (Swing the Left)
                SurfContactLeft = "left"
                movingDirection = np.random.choice(MovingDirectionList, 1)
                if movingDirection == "Y_positive" or movingDirection == "X_positive":  # Moving outwards need to be controlled
                    movePortion = np.random.uniform(0.0, 0.1)
                else:
                    movePortion = np.random.uniform(0.0, 1.0)
            elif surfNum % 2 == 1:  # Odd number of steps (Swing the Right)
                SurfContactLeft = "right"
                if movingDirection == "Y_negative" or movingDirection == "X_positive":  # Moving outwards need to be controlled
                    movePortion = np.random.uniform(0.0, 0.1)
                else:
                    movePortion = np.random.uniform(0.0, 1.0)

            # Move Patches
            movedSurf = patch_move_horizontal_percentage(surf=AllPatches[surfNum], PatchColumn=SurfContactLeft, SurfIndex=surfNum,
                                                         Direction=movingDirection, PercentageofMovingMargin=movePortion, AllSurfaces=AllPatches)
            AllPatches[surfNum] = movedSurf
            ContactSurfsVertice[surfNum-2] = movedSurf
            print("Move Contact Patch ", surfNum-2, "Horizontally along Direction of ",
                  movingDirection, "with portion of ", movePortion)

    # -----------------------------------------
    # Rotate the patches (can change rotation axis)
    # Clear All Patches (because it finishes its duty)


    #make rotation angle lists

    #for customized terrains
    if terrain_name == "customized":
        print("Rotating Patches with *Customized* Terrain")
        if isinstance(fixed_inclination, float) or (fixed_inclination == None): # if the fixed inclination is a single value (either flaot or None means random), we assign fixed incliation to all rotation patches, and all the rest (patch with type undefined) are 0
            print("fixed_inclination is either a fixed slope angle or None (for all rotated patches, all the rest are 0): ", fixed_inclination)
            surf_rot_angle = [fixed_inclination]*len(customized_terrain_pattern) + [0.0]*(len(ContactSurfsVertice)-len(customized_terrain_pattern))
        elif isinstance(fixed_inclination, list) and len(fixed_inclination) == 1: #if the fixed inclination is a single value (in list mode and None means random), we assign fixed incliation to all rotation patches, and all the rest (patch with type undefined) are 0
            print("fixed_inclination is either a fixed slope angle or None (for all rotated patches, all the rest are 0) (but in list mode): ", fixed_inclination)
            surf_rot_angle = fixed_inclination*len(customized_terrain_pattern) + + [0.0]*(len(ContactSurfsVertice)-len(customized_terrain_pattern))
        elif isinstance(fixed_inclination, list) and len(fixed_inclination) >= 1: #if the fixed inclination is a list with size >= 1
            if len(fixed_inclination) == len(customized_terrain_pattern): #list are not having equal size
                print("fixed_inclination is a list of pre-defined slope angles (consistent size with customized terrain pattern): ", fixed_inclination)
                surf_rot_angle = fixed_inclination + [0.0]*(len(ContactSurfsVertice)-len(customized_terrain_pattern)) #append 0 for the non rotating pathces (defined as flat in the pattern)
            else: #stop as we dont have same size of lists
                raise Exception("Size of fixed_inclination and customized_terrain_pattern is not equal")
    
    #for other types of terrains
    else:
        print("Rotating Patches with Other Types of Terrain: ", terrain_name)
        if isinstance(fixed_inclination, float) or (fixed_inclination == None): #if the fixed inclination is a single value (either flaot or None means random), we assign fixed incliation to all patches
            print("fixed_inclination is either a fixed slope angle or None: ", fixed_inclination)
            surf_rot_angle = [fixed_inclination]*len(ContactSurfsVertice)
        elif isinstance(fixed_inclination, list) and len(fixed_inclination) == 1: #if the fixed inclination is a single value (in list mode and None means random), we assign fixed incliation to all patches
            print("fixed_inclination is either a fixed slope angle or None (but in list mode): ", fixed_inclination)
            surf_rot_angle = fixed_inclination*len(ContactSurfsVertice)
        elif isinstance(fixed_inclination, list) and len(fixed_inclination) >= 1: #if the fixed inclination is a list with size >= 1
            print("fixed_inclination is a list of pre-defined slope angles (can be smaller size than the total number of patches): ", fixed_inclination)
            surf_rot_angle = fixed_inclination + [None]*(len(ContactSurfsVertice)-len(fixed_inclination)) #append None (random) for the patches with un-defined fixed-inclination

    AllPatches = []
    for ContactSurfNum in range(len(ContactSurfsVertice)):
        #If we enable the large slope to exist
        if large_slope_flag == True:
            # For Normal Patches
            if not (ContactSurfNum in large_slope_index):
                current_direction = terrain_pattern[ContactSurfNum]
                current_inclination = surf_rot_angle[ContactSurfNum]
            # For Large Slope Patches
            elif (ContactSurfNum in large_slope_index):
                # Get index in large_slope arrays
                idx_in_large_slope_array = large_slope_index.index(ContactSurfNum)
                current_direction = large_slope_directions[idx_in_large_slope_array]
                current_inclination = large_slope_inclinations[idx_in_large_slope_array]
        #No large slope enabled
        elif large_slope_flag == False:
            current_direction = terrain_pattern[ContactSurfNum]
            current_inclination = surf_rot_angle[ContactSurfNum]
        
        #rotate patches
        rotatedPatch = rotate_patch(
            surf=ContactSurfsVertice[ContactSurfNum], PatchType=current_direction, theta=current_inclination)

        ContactSurfsVertice[ContactSurfNum] = rotatedPatch
        
        #save current inclination
        if current_inclination != None and isinstance(current_inclination, float):
            ContactSurfsInclinationsDegrees.append(current_inclination/np.pi*180)
        elif current_inclination == None:
            ContactSurfsInclinationsDegrees.append(current_inclination)

        print("Rotate Patch ", ContactSurfNum, " along the direction of ", current_direction,
              "with inclination of ", current_inclination, "(None means Random)")

    # Update AllPatches with newly genreated ContactSurfVertices
    AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    # -------------------------------------------
    # Patch Elevation Changes (include non-change)
    for surfNum in range(2, TotalNumSurfs):
        # Compute elevation shift
        # For normal Patches
        if not ((surfNum-2) in large_slope_index):
            if randomElevationShift == True:
                elevation_dist = np.random.uniform(
                    min_elevation_shift, max_elevation_shift)
            else:
                elevation_dist = 0.0
        # For Large Slopes
        elif ((surfNum-2) in large_slope_index):
            idx_in_large_slope_array = large_slope_index.index(surfNum-2)
            elevation_dist = large_slope_Z_shifts[idx_in_large_slope_array]

        # manipulate the patch
        elevated_patch = elevation_shift_wrt_adjacent_patch(
            current_patch=AllPatches[surfNum], previous_patch=AllPatches[-1], elevation_shift=elevation_dist)

        print("Elevation change of Patch ", surfNum-2, "with ",
              elevation_dist, "with respect to the previous patch")

        # Rebuild the array
        AllPatches[surfNum] = elevated_patch
        ContactSurfsVertice[surfNum-2] = elevated_patch

    #------------------------------------------------
    # Shift the terrain with lab env elevation change
    if lab_blocks == True: #shift the surfaces if we use lab blocks
        print("lab blocks -> shift z to above horizontal line and shift the block z-axis with ", lab_block_z_shift)

        for surfNum in range(2, TotalNumSurfs): #loop over all the contact surfaces
            
            if terrain_pattern[surfNum-2] in ["X_positive", "X_negative", "Y_positive", "Y_negative", "DiagX_positive", "DiagX_negative", "DiagY_positive", "DiagY_negative"]: #only shift the stepping stone blocks
                #get curren patch
                cur_surf = AllPatches[surfNum]
                
                min_z_vertice = np.min([cur_surf[0,2], cur_surf[1,2], cur_surf[2,2], cur_surf[3,2]]) #get the lowest z vertice for all the blocks

                #lift all the points to have z >= 0 (min z = 0)
                cur_surf[0,2] = cur_surf[0,2] - min_z_vertice
                cur_surf[1,2] = cur_surf[1,2] - min_z_vertice
                cur_surf[2,2] = cur_surf[2,2] - min_z_vertice
                cur_surf[3,2] = cur_surf[3,2] - min_z_vertice

                #lift patches with the z offset (the small flat offset)
                cur_surf[0,2] = cur_surf[0,2] + lab_block_z_shift
                cur_surf[1,2] = cur_surf[1,2] + lab_block_z_shift
                cur_surf[2,2] = cur_surf[2,2] + lab_block_z_shift
                cur_surf[3,2] = cur_surf[3,2] + lab_block_z_shift

                if Constant_Block_Z_Shift_Flag == True:
                    cur_surf[0,2] = cur_surf[0,2] + Constant_Block_Z_Shift_Value
                    cur_surf[1,2] = cur_surf[1,2] + Constant_Block_Z_Shift_Value
                    cur_surf[2,2] = cur_surf[2,2] + Constant_Block_Z_Shift_Value
                    cur_surf[3,2] = cur_surf[3,2] + Constant_Block_Z_Shift_Value

                #Rebuild the contact patch array
                AllPatches[surfNum] = cur_surf
                ContactSurfsVertice[surfNum-2] = cur_surf

    # ------------
    # Build Useful arrays (Only for Contact Surfaces, NOTE the index)
    #   Build containers
    ContactSurfsHalfSpace = []
    ContactSurfsTypes = []
    ContactSurfsTangentX = []
    ContactSurfsTangentY = []
    ContactSurfsNorm = []
    ContactSurfsOrientation = []

    #   Get useful arrays
    for ContactSurfNum in range(len(ContactSurfsVertice)):
        HalfSpace_temp = np.concatenate(convert_surface_to_inequality(
            ContactSurfsVertice[ContactSurfNum].T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            ContactSurfsVertice[ContactSurfNum])

        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsTypes.append(terrain_pattern[ContactSurfNum])
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

    # Build All Patches array
    AllPatches = [Sl0, Sr0] + ContactSurfsVertice
    
    # Build Terrain Model Vector
    TerrainModel = {"InitLeftSurfVertice": Sl0,  "InitLeftSurfType": LeftSurfType,
                    "InitLeftSurfTangentX": Sl0_TangentX, "InitLeftSurfTangentY": Sl0_TangentY, "InitLeftSurfNorm": Sl0_Norm, "InitLeftSurfOrientation": Sl0_Orientation,
                    "InitRightSurfVertice": Sr0, "InitRightSurfType": RightSurfType,
                    "InitRightSurfTangentX": Sr0_TangentX, "InitRightSurfTangentY": Sr0_TangentY, "InitRightSurfNorm": Sr0_Norm, "InitRightSurfOrientation": Sr0_Orientation,
                    "ContactSurfsVertice": ContactSurfsVertice,
                    "ContactSurfsHalfSpace": ContactSurfsHalfSpace,
                    "ContactSurfsTypes": ContactSurfsTypes,
                    "ContactSurfsNames": ContactSurfsNames,
                    "ContactSurfsInclinationsDegrees": ContactSurfsInclinationsDegrees,
                    "ContactSurfsTangentX": ContactSurfsTangentX,
                    "ContactSurfsTangentY": ContactSurfsTangentY,
                    "ContactSurfsNorm": ContactSurfsNorm,
                    "ContactSurfsOrientation": ContactSurfsOrientation,
                    "AllPatchesVertices": AllPatches}

    #contact sequence get doubled as we want to put two steps on a single patch 
    #   Build containers
    Doubled_ContactSurfsVertice = []
    Doubled_ContactSurfsHalfSpace = []
    Doubled_ContactSurfsTypes = []
    Doubled_ContactSurfsNames = []
    Doubled_ContactSurfsInclinationsDegrees = []
    Doubled_ContactSurfsTangentX = []
    Doubled_ContactSurfsTangentY = []
    Doubled_ContactSurfsNorm = []
    Doubled_ContactSurfsOrientation = []
    if twosteps_on_patch == True:
        for patch_index in range(len(ContactSurfsVertice)):
            if patch_index%2 == 0: #even number, then double the patches
                Doubled_ContactSurfsVertice.extend(ContactSurfsVertice[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsHalfSpace.extend(ContactSurfsHalfSpace[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsTypes.extend(ContactSurfsTypes[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsNames.extend(ContactSurfsNames[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsInclinationsDegrees.extend(ContactSurfsInclinationsDegrees[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsTangentX.extend(ContactSurfsTangentX[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsTangentY.extend(ContactSurfsTangentY[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsNorm.extend(ContactSurfsNorm[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsOrientation.extend(ContactSurfsOrientation[patch_index:patch_index+1+1]*2)

    if inner_blocks == True:
        for patch_index in range(len(ContactSurfsVertice)-1): #in case we have odd number of patches
            if patch_index%2 == 0: #even number, then double the patches
                #print("!!!!!Patch Index",patch_index)
                Temp_Patch1_Vertice = copy.deepcopy(ContactSurfsVertice[patch_index])
                Temp_Patch1_Vertice = cut_patch_from_one_boder(patch = Temp_Patch1_Vertice, rotation_type = ContactSurfsTypes[patch_index], rotation_angle =ContactSurfsInclinationsDegrees[patch_index]/180*np.pi, cut_border_side = 'right', cut_length = np.abs(Proj_Length-inner_block_length))
                Temp_Patch1_HalfSpce = np.concatenate(convert_surface_to_inequality(Temp_Patch1_Vertice.T), axis=None)

                Temp_Patch2_Vertice = copy.deepcopy(ContactSurfsVertice[patch_index+1])
                Temp_Patch2_Vertice = cut_patch_from_one_boder(patch= Temp_Patch2_Vertice, rotation_type = ContactSurfsTypes[patch_index+1], rotation_angle = ContactSurfsInclinationsDegrees[patch_index+1]/180*np.pi, cut_border_side = 'right', cut_length = np.abs(Proj_Length-inner_block_length))
                Temp_Patch2_HalfSpce = np.concatenate(convert_surface_to_inequality(Temp_Patch2_Vertice.T), axis=None)

                Temp_Patch3_Vertice = copy.deepcopy(ContactSurfsVertice[patch_index])
                Temp_Patch3_Vertice = cut_patch_from_one_boder(patch = Temp_Patch3_Vertice, rotation_type = ContactSurfsTypes[patch_index], rotation_angle = ContactSurfsInclinationsDegrees[patch_index]/180*np.pi, cut_border_side = 'left', cut_length = np.abs(Proj_Length-inner_block_length))
                Temp_Patch3_HalfSpce = np.concatenate(convert_surface_to_inequality(Temp_Patch3_Vertice.T), axis=None)

                Temp_Patch4_Vertice = copy.deepcopy(ContactSurfsVertice[patch_index+1])
                Temp_Patch4_Vertice = cut_patch_from_one_boder(patch= Temp_Patch4_Vertice, rotation_type = ContactSurfsTypes[patch_index+1], rotation_angle = ContactSurfsInclinationsDegrees[patch_index+1]/180*np.pi, cut_border_side = 'left', cut_length = np.abs(Proj_Length-inner_block_length))
                Temp_Patch4_HalfSpce = np.concatenate(convert_surface_to_inequality(Temp_Patch4_Vertice.T), axis=None)

                Doubled_ContactSurfsVertice.extend([Temp_Patch1_Vertice,Temp_Patch2_Vertice,Temp_Patch3_Vertice,Temp_Patch4_Vertice])
                Doubled_ContactSurfsHalfSpace.extend([Temp_Patch1_HalfSpce,Temp_Patch2_HalfSpce,Temp_Patch3_HalfSpce,Temp_Patch4_HalfSpce])

                Doubled_ContactSurfsTypes.extend(ContactSurfsTypes[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsNames.extend(ContactSurfsNames[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsInclinationsDegrees.extend(ContactSurfsInclinationsDegrees[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsTangentX.extend(ContactSurfsTangentX[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsTangentY.extend(ContactSurfsTangentY[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsNorm.extend(ContactSurfsNorm[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsOrientation.extend(ContactSurfsOrientation[patch_index:patch_index+1+1]*2)
    # ------------

    #AllPatches = [Sl0, Sr0] + Doubled_ContactSurfsVertice

    # ------------
    #update the terrain model
    if twosteps_on_patch == True or inner_blocks == True:
        TerrainModel["ContactSurfsVertice"] = Doubled_ContactSurfsVertice
        TerrainModel["ContactSurfsHalfSpace"] = Doubled_ContactSurfsHalfSpace
        TerrainModel["ContactSurfsTypes"] = Doubled_ContactSurfsTypes
        TerrainModel["ContactSurfsNames"] = Doubled_ContactSurfsNames
        TerrainModel["ContactSurfsInclinationsDegrees"] = Doubled_ContactSurfsInclinationsDegrees
        TerrainModel["ContactSurfsTangentX"] = Doubled_ContactSurfsTangentX
        TerrainModel["ContactSurfsTangentY"] = Doubled_ContactSurfsTangentY
        TerrainModel["ContactSurfsNorm"] = Doubled_ContactSurfsNorm
        TerrainModel["ContactSurfsOrientation"] = Doubled_ContactSurfsOrientation

    print("Contact Surfaces inclinations: ", ContactSurfsInclinationsDegrees)

    #print("length!!!:",len(ContactSurfsVertice))
    #print("length!!!:",len(TerrainModel["ContactSurfsTangentX"]))

    return TerrainModel


#The one we will be using for lab experiments
def terrain_model_gen_lab(terrain_name=None,
                      customized_terrain_pattern = [],
                      Proj_Length=0.6, Proj_Width=0.6,
                      fixed_inclination=0.0,
                      lab_blocks = False, lab_block_z_shift = 0.0,
                      randomInitSurfSize=False,
                      random_surfsize=False, min_shrink_factor=0.0, max_shrink_factor=0.3,
                      randomHorizontalMove=False,  # Need to add random Height Move for Normal Patches
                      randomElevationShift=False, min_elevation_shift=-0.075, max_elevation_shift=0.075,
                      randomMisAlignmentofFirstTwoPatches=False, MisAlignmentColumn=None, MisAlignmentAmount=0.25,
                      gap_between_patches = False,
                      x_gap = 0.0,
                      NumSteps=None, NumLookAhead=None,
                      large_slope_flag=False, large_slope_index=[],
                      large_slope_directions=[], large_slope_inclinations=[],
                      large_slope_X_shifts=[], large_slope_Y_shifts=[], large_slope_Z_shifts=[],
                      y_center = 0.0,
                      x_offset = 0.0,
                      twosteps_on_patch = False):


    # Check large slope parameters are in the same length
    if large_slope_flag == True:
        if (len(large_slope_index) == len(large_slope_directions)) and (len(large_slope_directions) == len(large_slope_inclinations)) and \
           (len(large_slope_inclinations) == len(large_slope_X_shifts)) and (len(large_slope_X_shifts) == len(large_slope_Y_shifts)) and \
           (len(large_slope_Y_shifts) == len(large_slope_Z_shifts)):
            print("All Large Slope Parameters in the same Length")
        else:
            raise Exception("Large Slope Parameters in a Different Length")

    # Surf Vertex Identification
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4

    # ---------------
    # Decide Parameters
    # Number of Surfaces based on Number of steps and Number of lookahead and initial patches
    TotalNumSurfs = NumSteps + NumLookAhead - 1 + 2  # (include the intial two contact patches

    # ---------------
    # Decide Which column will have Mis Alignment of (the first two patches) if not provided
    if randomMisAlignmentofFirstTwoPatches == True:
        if MisAlignmentColumn == None:
            MisAlignmentColumn = np.random.choice(["left", "right"], 1)
    elif randomMisAlignmentofFirstTwoPatches == False:
        MisAlignmentColumn = "left"

    # Convert to the index of the mismatch (NOTE: idx is for COntact Patch sequence, not all patches; start from zero)
    if MisAlignmentColumn == "left":
        MisAlignedPatchIdx = 0
    elif MisAlignmentColumn == "right":
        MisAlignedPatchIdx = 1
    else:
        raise Exception("Unknown column name")

    # ---------------
    # Generate Initial Patches (Currently Define as flat patches)
    if randomInitSurfSize == False:
        InitContactSurf_x_max = 0.15 + x_offset
    elif randomInitSurfSize == True:
        InitContactSurf_x_max = np.random.uniform(0.115, 0.215) + x_offset
    else:
        raise Exception("Unknow Flag for Init Contact Surf Left most Boder")

    print("Left Init Contact Surface:")
    Sl0 = np.array([[InitContactSurf_x_max, 1.0+y_center, 0.], [-0.2, 1.0+y_center, 0.],
                   [-0.2, y_center, 0.], [InitContactSurf_x_max, y_center, 0.0]])
    #Sl0 = np.array([[0.35, 1.0, 0.], [-0.35, 1.0, 0.], [-0.35, 0.0, 0.], [0.35, 0.0, 0.0]])
    LeftSurfType = getSurfaceType(Sl0)  # "flat"
    print("Left Init Contact Surface Type: ", LeftSurfType)
    Sl0_TangentX, Sl0_TangentY, Sl0_Norm, Sl0_Orientation = getTerrainTagentsNormOrientation(
        Sl0)

    print("Right Init Contact Surface")
    Sr0 = np.array([[InitContactSurf_x_max, y_center, 0.], [-0.2, y_center, 0.],
                   [-0.2, -1.0 + y_center, 0.], [InitContactSurf_x_max, -1.0 + y_center, 0.0]])
    #Sr0 = np.array([[0.7, 0, 0.], [0.0, 0, 0.], [0.0, -1.0, 0.], [0.7, -1.0, 0.0]])
    RightSurfType = getSurfaceType(Sr0)  # "flat"
    print("Right Init Contact Surface Type: ", RightSurfType)
    Sr0_TangentX, Sr0_TangentY, Sr0_Norm, Sr0_Orientation = getTerrainTagentsNormOrientation(
        Sr0)

    # ------------
    # Make Contact Patches
    # --------------------------
    #   Decide Terrain Pattern
    if (terrain_name == "flat") or (terrain_name == "stair") or (terrain_name == "single_flat"):
        terrain_pattern = ["flat", "flat", "flat", "flat"] * 50  # 200 steps

    elif terrain_name == "antfarm_left":
        # up v first, then down v
        # NOTE: Currently Assume the Initial Contacts are Flat
        terrain_pattern = ["X_negative", "X_positive",
                           "X_positive", "X_negative"] * 50  # 200 steps

    elif terrain_name == "up_and_down_left":
        terrain_pattern = ["Y_negative", "Y_negative",
                           "Y_positive", "Y_positive"] * 50  # 200 steps

    elif terrain_name == "random":
        #TerrainTypeList = ["flat", "X_positive", "X_negative", "Y_positive", "Y_negative"]
        # 4 types of terrain, no flat
        TerrainTypeList = ["X_positive",
                           "X_negative", "Y_positive", "Y_negative"]
        terrain_pattern = np.random.choice(TerrainTypeList, 200)
    elif terrain_name == "random_diag":
        TerrainTypeList = ["X_positive","X_negative", "Y_positive", "Y_negative",
                           "DiagX_positive", "DiagX_negative", "DiagY_positive", "DiagY_negative"]
        terrain_pattern = np.random.choice(TerrainTypeList, 200)
    elif terrain_name == "customized":
        terrain_pattern = customized_terrain_pattern + ["flat"]*200
    else:
        raise Exception("Unknown Terrain Type")

    print(terrain_pattern)

    # -----------------------------------
    #   Create container for Patches
    ContactSurfsVertice = []
    ContactSurfsTypes = []
    ContactSurfsNames = []
    ContactSurfsInclinationsDegrees = []
    AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    # ------------------------------------
    # Generate a sequence of flat patches
    for surfNum in range(2, TotalNumSurfs):
        print("Put Contact Patch ", surfNum)

        # Decide Coordinate of reference point
        if surfNum % 2 == 0:  # Even number of steps (Swing the Left)
            # ----       #p2---------------------p1
            #            |                      |
            # Sn-2(Sl0) # |         Sn           |
            # (L) #      |        (L)           |
            # ----      #p3 (ref_x, ref_y)-------p4
            #            |                      |
            # Sn-1(Sr0) # |                      |
            # (R) #      |                      |

            # Define which Contact foot is for the Surface
            SurfContactLeft = "left"
            # ref_x decided by the P1_x of Sn-2 (-2 in index)
            ref_x = AllPatches[-2][0][0]
            # ref_y decided by the P1_y of Sn-1 (-1 in index)
            ref_y = AllPatches[-1][0][1]
            #center = getCenter(AllPatches[-1])
            # center[2] #ref_z decided by the central z of Sn-1 (relative foot height constraints is enforced with respect to stationary foot)
            ref_z = 0.0

        elif surfNum % 2 == 1:  # Odd number of steps
            # ---- #        ---------------------
            #      |                      |
            #      |         Sn-1         |
            #      |          (L)         |
            # ---- #      ---------------------
            #     P2 (ref_x, ref_y)------P1
            #      |                      |
            # Sn-2(Sr0) # |         Sn           |
            # (R) #      |        (R)           |
            #    P3---------------------P4

            # Define which Contact foot is for the Surface
            SurfContactLeft = "right"
            ref_x = AllPatches[-2][0][0]  # ref_x decided by the P1_x of Sn-2
            # ref_y decided by the P3_y of Sn-1 (ContactSurfsVertice[-1])
            ref_y = AllPatches[-1][2][1]
            #center = getCenter(AllPatches[-1])
            # center[2] #ref_z decided by the central z of Sn-1 (relative foot height constraints is enforced with respect to stationary foot)
            ref_z = 0.0

        # Shift along x-direction for mis alignment of First Two Patches
        # Convert from Contact Patch Sequence index (start from 0) to all patches index (+ 2)
        if surfNum == MisAlignedPatchIdx + 2:
            if randomMisAlignmentofFirstTwoPatches == True:  # if we randomly mis align the first two patches
                # MisAlignmentAmount = np.random.uniform(0.0,MaxMisAllignmentAmount) #random choose a mis aligned amount
                ref_x = ref_x + MisAlignmentAmount  # shift the x-ref of the selected patch

        #Give gaps to the patches
        if gap_between_patches == True:
            if surfNum > 2: #we give gap starting from the 4th patch (right column)
                ref_x = ref_x + x_gap

        # Build an initial flat surface
        surf_temp = flat_patch_gen(PatchColumn=SurfContactLeft, ref_x=ref_x,
                                   ref_y=ref_y, ref_z=ref_z, Proj_Length=Proj_Length, Proj_Width=Proj_Width)

        # Add to current surface to lists
        ContactSurfsVertice.append(surf_temp)
        ContactSurfsNames.append("S"+str(surfNum-2))
        ContactSurfsTypes.append(terrain_pattern[surfNum])
        AllPatches.append(surf_temp)

    # -----------------------------------
    # Modify the patches
    # Clear All Patches (because it finishes its duty)
    AllPatches = []
    # ----------------------------------------
    # (Randomly) Shrink size if we want (only make smaller)
    if random_surfsize == True:
        for ContactSurfNum in range(len(ContactSurfsVertice)):
            shrinkFactor = np.random.uniform(
                min_shrink_factor, max_shrink_factor)
            ContactSurfsVertice[ContactSurfNum] = flat_patch_shrink(
                surf=ContactSurfsVertice[ContactSurfNum], shrinkFactor=shrinkFactor, Proj_Length=Proj_Length, Proj_Width=Proj_Width)
            print("Shrink (Smaller) Contact Factor of Patch ",
                  ContactSurfNum, "with Factor of ", shrinkFactor)

    # Update AllPatches with newly genreated ContactSurfVertices
    AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    # -----------------------------------------------------------
    # Move the patches (in X Y directions only) (NOTE: We need to ensure no over shoot)
    if randomHorizontalMove == True:
        MovingDirectionList = ["X_positive",
                               "X_negative", "Y_positive", "Y_negative"]
        for surfNum in range(2, TotalNumSurfs):
            # Decide Coordinate of reference point
            if surfNum % 2 == 0:  # Even number of steps (Swing the Left)
                SurfContactLeft = "left"
                movingDirection = np.random.choice(MovingDirectionList, 1)
                if movingDirection == "Y_positive" or movingDirection == "X_positive":  # Moving outwards need to be controlled
                    movePortion = np.random.uniform(0.0, 0.1)
                else:
                    movePortion = np.random.uniform(0.0, 1.0)
            elif surfNum % 2 == 1:  # Odd number of steps (Swing the Right)
                SurfContactLeft = "right"
                if movingDirection == "Y_negative" or movingDirection == "X_positive":  # Moving outwards need to be controlled
                    movePortion = np.random.uniform(0.0, 0.1)
                else:
                    movePortion = np.random.uniform(0.0, 1.0)

            # Move Patches
            movedSurf = patch_move_horizontal_percentage(surf=AllPatches[surfNum], PatchColumn=SurfContactLeft, SurfIndex=surfNum,
                                                         Direction=movingDirection, PercentageofMovingMargin=movePortion, AllSurfaces=AllPatches)
            AllPatches[surfNum] = movedSurf
            ContactSurfsVertice[surfNum-2] = movedSurf
            print("Move Contact Patch ", surfNum-2, "Horizontally along Direction of ",
                  movingDirection, "with portion of ", movePortion)

    # -----------------------------------------
    # Rotate the patches (can change rotation axis)
    # Clear All Patches (because it finishes its duty)


    #make rotation angle lists

    #for customized terrains
    if terrain_name == "customized":
        print("Rotating Patches with *Customized* Terrain")
        if isinstance(fixed_inclination, float) or (fixed_inclination == None): # if the fixed inclination is a single value (either flaot or None means random), we assign fixed incliation to all rotation patches, and all the rest (patch with type undefined) are 0
            print("fixed_inclination is either a fixed slope angle or None (for all rotated patches, all the rest are 0): ", fixed_inclination)
            surf_rot_angle = [fixed_inclination]*len(customized_terrain_pattern) + [0.0]*(len(ContactSurfsVertice)-len(customized_terrain_pattern))
        elif isinstance(fixed_inclination, list) and len(fixed_inclination) == 1: #if the fixed inclination is a single value (in list mode and None means random), we assign fixed incliation to all rotation patches, and all the rest (patch with type undefined) are 0
            print("fixed_inclination is either a fixed slope angle or None (for all rotated patches, all the rest are 0) (but in list mode): ", fixed_inclination)
            surf_rot_angle = fixed_inclination*len(customized_terrain_pattern) + + [0.0]*(len(ContactSurfsVertice)-len(customized_terrain_pattern))
        elif isinstance(fixed_inclination, list) and len(fixed_inclination) >= 1: #if the fixed inclination is a list with size >= 1
            if len(fixed_inclination) == len(customized_terrain_pattern): #list are not having equal size
                print("fixed_inclination is a list of pre-defined slope angles (consistent size with customized terrain pattern): ", fixed_inclination)
                surf_rot_angle = fixed_inclination + [0.0]*(len(ContactSurfsVertice)-len(customized_terrain_pattern)) #append 0 for the non rotating pathces (defined as flat in the pattern)
            else: #stop as we dont have same size of lists
                raise Exception("Size of fixed_inclination and customized_terrain_pattern is not equal")
    
    #for other types of terrains
    else:
        print("Rotating Patches with Other Types of Terrain: ", terrain_name)
        if isinstance(fixed_inclination, float) or (fixed_inclination == None): #if the fixed inclination is a single value (either flaot or None means random), we assign fixed incliation to all patches
            print("fixed_inclination is either a fixed slope angle or None: ", fixed_inclination)
            surf_rot_angle = [fixed_inclination]*len(ContactSurfsVertice)
        elif isinstance(fixed_inclination, list) and len(fixed_inclination) == 1: #if the fixed inclination is a single value (in list mode and None means random), we assign fixed incliation to all patches
            print("fixed_inclination is either a fixed slope angle or None (but in list mode): ", fixed_inclination)
            surf_rot_angle = fixed_inclination*len(ContactSurfsVertice)
        elif isinstance(fixed_inclination, list) and len(fixed_inclination) >= 1: #if the fixed inclination is a list with size >= 1
            print("fixed_inclination is a list of pre-defined slope angles (can be smaller size than the total number of patches): ", fixed_inclination)
            surf_rot_angle = fixed_inclination + [None]*(len(ContactSurfsVertice)-len(fixed_inclination)) #append None (random) for the patches with un-defined fixed-inclination

    AllPatches = []
    for ContactSurfNum in range(len(ContactSurfsVertice)):
        #If we enable the large slope to exist
        if large_slope_flag == True:
            # For Normal Patches
            if not (ContactSurfNum in large_slope_index):
                current_direction = terrain_pattern[ContactSurfNum]
                current_inclination = surf_rot_angle[ContactSurfNum]
            # For Large Slope Patches
            elif (ContactSurfNum in large_slope_index):
                # Get index in large_slope arrays
                idx_in_large_slope_array = large_slope_index.index(ContactSurfNum)
                current_direction = large_slope_directions[idx_in_large_slope_array]
                current_inclination = large_slope_inclinations[idx_in_large_slope_array]
        #No large slope enabled
        elif large_slope_flag == False:
            current_direction = terrain_pattern[ContactSurfNum]
            current_inclination = surf_rot_angle[ContactSurfNum]
        
        #rotate patches
        rotatedPatch = rotate_patch(
            surf=ContactSurfsVertice[ContactSurfNum], PatchType=current_direction, theta=current_inclination)
        
        ContactSurfsVertice[ContactSurfNum] = rotatedPatch
        
        #save current inclination
        if current_inclination != None and isinstance(current_inclination, float):
            ContactSurfsInclinationsDegrees.append(current_inclination/np.pi*180)
        elif current_inclination == None:
            ContactSurfsInclinationsDegrees.append(current_inclination)

        print("Rotate Patch ", ContactSurfNum, " along the direction of ", current_direction,
              "with inclination of ", current_inclination, "(None means Random)")

    # Update AllPatches with newly genreated ContactSurfVertices
    AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    # -------------------------------------------
    # Patch Elevation Changes (include non-change)
    for surfNum in range(2, TotalNumSurfs):
        # Compute elevation shift
        # For normal Patches
        if not ((surfNum-2) in large_slope_index):
            if randomElevationShift == True:
                elevation_dist = np.random.uniform(
                    min_elevation_shift, max_elevation_shift)
            else:
                elevation_dist = 0.0
        # For Large Slopes
        elif ((surfNum-2) in large_slope_index):
            idx_in_large_slope_array = large_slope_index.index(surfNum-2)
            elevation_dist = large_slope_Z_shifts[idx_in_large_slope_array]

        # manipulate the patch
        elevated_patch = elevation_shift_wrt_adjacent_patch(
            current_patch=AllPatches[surfNum], previous_patch=AllPatches[-1], elevation_shift=elevation_dist)

        print("Elevation change of Patch ", surfNum-2, "with ",
              elevation_dist, "with respect to the previous patch")

        # Rebuild the array
        AllPatches[surfNum] = elevated_patch
        ContactSurfsVertice[surfNum-2] = elevated_patch

    #------------------------------------------------
    # Shift the terrain with lab env elevation change
    if lab_blocks == True: #shift the surfaces if we use lab blocks
        print("lab blocks -> shift z to above horizontal line and shift the block z-axis with ", lab_block_z_shift)

        for surfNum in range(2, TotalNumSurfs): #loop over all the contact surfaces
            
            if terrain_pattern[surfNum-2] in ["X_positive", "X_negative", "Y_positive", "Y_negative", "DiagX_positive", "DiagX_negative", "DiagY_positive", "DiagY_negative"]: #only shift the stepping stone blocks
                #get curren patch
                cur_surf = AllPatches[surfNum]
                
                min_z_vertice = np.min([cur_surf[0,2], cur_surf[1,2], cur_surf[2,2], cur_surf[3,2]]) #get the lowest z vertice for all the blocks

                #lift all the points to have z >= 0 (min z = 0)
                cur_surf[0,2] = cur_surf[0,2] - min_z_vertice
                cur_surf[1,2] = cur_surf[1,2] - min_z_vertice
                cur_surf[2,2] = cur_surf[2,2] - min_z_vertice
                cur_surf[3,2] = cur_surf[3,2] - min_z_vertice

                #lift patches with the z offset (the small flat offset)
                cur_surf[0,2] = cur_surf[0,2] + lab_block_z_shift
                cur_surf[1,2] = cur_surf[1,2] + lab_block_z_shift
                cur_surf[2,2] = cur_surf[2,2] + lab_block_z_shift
                cur_surf[3,2] = cur_surf[3,2] + lab_block_z_shift

                #Rebuild the contact patch array
                AllPatches[surfNum] = cur_surf
                ContactSurfsVertice[surfNum-2] = cur_surf

    # ------------
    # Build Useful arrays (Only for Contact Surfaces, NOTE the index)
    #   Build containers
    ContactSurfsHalfSpace = []
    ContactSurfsTypes = []
    ContactSurfsTangentX = []
    ContactSurfsTangentY = []
    ContactSurfsNorm = []
    ContactSurfsOrientation = []

    #   Get useful arrays
    for ContactSurfNum in range(len(ContactSurfsVertice)):
        HalfSpace_temp = np.concatenate(convert_surface_to_inequality(
            ContactSurfsVertice[ContactSurfNum].T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            ContactSurfsVertice[ContactSurfNum])

        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsTypes.append(terrain_pattern[ContactSurfNum])
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

    # Build All Patches array
    AllPatches = [Sl0, Sr0] + ContactSurfsVertice
    
    #contact sequence get doubled as we want to put two steps on a single patch 
    #   Build containers
    Doubled_ContactSurfsVertice = []
    Doubled_ContactSurfsHalfSpace = []
    Doubled_ContactSurfsTypes = []
    Doubled_ContactSurfsNames = []
    Doubled_ContactSurfsInclinationsDegrees = []
    Doubled_ContactSurfsTangentX = []
    Doubled_ContactSurfsTangentY = []
    Doubled_ContactSurfsNorm = []
    Doubled_ContactSurfsOrientation = []
    if twosteps_on_patch == True:
        for patch_index in range(len(ContactSurfsVertice)):
            if patch_index%2 == 0: #even number, then double the patches
                Doubled_ContactSurfsVertice.extend(ContactSurfsVertice[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsHalfSpace.extend(ContactSurfsHalfSpace[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsTypes.extend(ContactSurfsTypes[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsNames.extend(ContactSurfsNames[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsInclinationsDegrees.extend(ContactSurfsInclinationsDegrees[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsTangentX.extend(ContactSurfsTangentX[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsTangentY.extend(ContactSurfsTangentY[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsNorm.extend(ContactSurfsNorm[patch_index:patch_index+1+1]*2)
                Doubled_ContactSurfsOrientation.extend(ContactSurfsOrientation[patch_index:patch_index+1+1]*2)

    # ------------
    # Build Terrain Model Vector
    TerrainModel = {"InitLeftSurfVertice": Sl0,  "InitLeftSurfType": LeftSurfType,
                    "InitLeftSurfTangentX": Sl0_TangentX, "InitLeftSurfTangentY": Sl0_TangentY, "InitLeftSurfNorm": Sl0_Norm, "InitLeftSurfOrientation": Sl0_Orientation,
                    "InitRightSurfVertice": Sr0, "InitRightSurfType": RightSurfType,
                    "InitRightSurfTangentX": Sr0_TangentX, "InitRightSurfTangentY": Sr0_TangentY, "InitRightSurfNorm": Sr0_Norm, "InitRightSurfOrientation": Sr0_Orientation,
                    "ContactSurfsVertice": ContactSurfsVertice,
                    "ContactSurfsHalfSpace": ContactSurfsHalfSpace,
                    "ContactSurfsTypes": ContactSurfsTypes,
                    "ContactSurfsNames": ContactSurfsNames,
                    "ContactSurfsInclinationsDegrees": ContactSurfsInclinationsDegrees,
                    "ContactSurfsTangentX": ContactSurfsTangentX,
                    "ContactSurfsTangentY": ContactSurfsTangentY,
                    "ContactSurfsNorm": ContactSurfsNorm,
                    "ContactSurfsOrientation": ContactSurfsOrientation,
                    "AllPatchesVertices": AllPatches}

    #update the terrain model
    if twosteps_on_patch == True:
        TerrainModel["ContactSurfsVertice"] = Doubled_ContactSurfsVertice
        TerrainModel["ContactSurfsHalfSpace"] = Doubled_ContactSurfsHalfSpace
        TerrainModel["ContactSurfsTypes"] = Doubled_ContactSurfsTypes
        TerrainModel["ContactSurfsNames"] = Doubled_ContactSurfsNames
        TerrainModel["ContactSurfsInclinationsDegrees"] = Doubled_ContactSurfsInclinationsDegrees
        TerrainModel["ContactSurfsTangentX"] = Doubled_ContactSurfsTangentX
        TerrainModel["ContactSurfsTangentY"] = Doubled_ContactSurfsTangentY
        TerrainModel["ContactSurfsNorm"] = Doubled_ContactSurfsNorm
        TerrainModel["ContactSurfsOrientation"] = Doubled_ContactSurfsOrientation

    print("Contact Surfaces inclinations: ", ContactSurfsInclinationsDegrees)

    #print("length!!!:",len(ContactSurfsVertice))
    #print("length!!!:",len(TerrainModel["ContactSurfsTangentX"]))

    return TerrainModel

#The one we used for simulation computation
def terrain_model_gen(terrain_name=None,
                      Proj_Length=0.6, Proj_Width=0.6,
                      fixed_inclination=0.0,
                      randomInitSurfSize=False,
                      random_surfsize=False, min_shrink_factor=0.0, max_shrink_factor=0.3,
                      randomHorizontalMove=False,  # Need to add random Height Move for Normal Patches
                      randomElevationShift=False, min_elevation_shift=-0.075, max_elevation_shift=0.075,
                      randomMisAlignmentofFirstTwoPatches=False, MisAlignmentColumn=None, MisAlignmentAmount=0.25,
                      gap_between_patches = False,
                      x_gap = 0.0,
                      NumSteps=None, NumLookAhead=None,
                      large_slope_flag=False, large_slope_index=[8],
                      large_slope_directions=["X_positive"], large_slope_inclinations=[18.0/180.0*np.pi],
                      large_slope_X_shifts=[0.0], large_slope_Y_shifts=[0.0], large_slope_Z_shifts=[0.0],
                      y_center = 0.0,
                      x_offset = 0.0):

    print("y_center: ",y_center)
    print("x_offset: ",x_offset)

    # Check large slope parameters are in the same length
    if large_slope_flag == True:
        if (len(large_slope_index) == len(large_slope_directions)) and (len(large_slope_directions) == len(large_slope_inclinations)) and \
           (len(large_slope_inclinations) == len(large_slope_X_shifts)) and (len(large_slope_X_shifts) == len(large_slope_Y_shifts)) and \
           (len(large_slope_Y_shifts) == len(large_slope_Z_shifts)):
            print("All Large Slope Parameters in the same Length")
        else:
            raise Exception("Large Slope Parameters in a Different Length")

    # Surf Vertex Identification
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4

    # ---------------
    # Decide Parameters
    # Number of Surfaces based on Number of steps and Number of lookahead and initial patches
    TotalNumSurfs = NumSteps + NumLookAhead - 1 + 2  # (include the intial two contact patches)

    # ---------------
    # Decide Which column will have Mis Alignment of in (the first two patches) if not provided
    if randomMisAlignmentofFirstTwoPatches == True:
        if MisAlignmentColumn == None:
            MisAlignmentColumn = np.random.choice(["left", "right"], 1)
    elif randomMisAlignmentofFirstTwoPatches == False:
        MisAlignmentColumn = "left"

    # Convert to the index of the mismatch (NOTE: idx is for COntact Patch sequence, not all patches; start from zero)
    if MisAlignmentColumn == "left":
        MisAlignedPatchIdx = 0
    elif MisAlignmentColumn == "right":
        MisAlignedPatchIdx = 1
    else:
        raise Exception("Unknown column name")

    # ---------------
    # Generate Initial Patches (Currently Define as flat patches)
    if randomInitSurfSize == False:
        InitContactSurf_x_max = 0.15 + x_offset  #0.15
    elif randomInitSurfSize == True:
        InitContactSurf_x_max = np.random.uniform(0.115, 0.215) + x_offset
    else:
        raise Exception("Unknow Flag for Init Contact Surf Left most Boder")

    print("Left Init Contact Surface:")
    Sl0 = np.array([[InitContactSurf_x_max, 1.0+y_center, 0.], [-0.2, 1.0+y_center, 0.],
                   [-0.2, y_center, 0.], [InitContactSurf_x_max, y_center, 0.0]])
    #Sl0 = np.array([[0.35, 1.0, 0.], [-0.35, 1.0, 0.], [-0.35, 0.0, 0.], [0.35, 0.0, 0.0]])
    LeftSurfType = getSurfaceType(Sl0)  # "flat"
    print("Left Init Contact Surface Type: ", LeftSurfType)
    Sl0_TangentX, Sl0_TangentY, Sl0_Norm, Sl0_Orientation = getTerrainTagentsNormOrientation(
        Sl0)

    print("Right Init Contact Surface")
    Sr0 = np.array([[InitContactSurf_x_max, y_center, 0.], [-0.2, y_center, 0.],
                   [-0.2, -1.0 + y_center, 0.], [InitContactSurf_x_max, -1.0 + y_center, 0.0]])
    #Sr0 = np.array([[0.7, 0, 0.], [0.0, 0, 0.], [0.0, -1.0, 0.], [0.7, -1.0, 0.0]])
    RightSurfType = getSurfaceType(Sr0)  # "flat"
    print("Right Init Contact Surface Type: ", RightSurfType)
    Sr0_TangentX, Sr0_TangentY, Sr0_Norm, Sr0_Orientation = getTerrainTagentsNormOrientation(
        Sr0)

    # ------------
    # Make Contact Patches
    # --------------------------
    #   Decide Terrain Pattern
    if (terrain_name == "flat") or (terrain_name == "stair") or (terrain_name == "single_flat"):
        terrain_pattern = ["flat", "flat", "flat", "flat"] * 50  # 200 steps

    elif terrain_name == "antfarm_left":
        # up v first, then down v
        # NOTE: Currently Assume the Initial Contacts are Flat
        terrain_pattern = ["X_negative", "X_positive",
                           "X_positive", "X_negative"] * 50  # 200 steps

    elif terrain_name == "up_and_down_left":
        terrain_pattern = ["Y_negative", "Y_negative",
                           "Y_positive", "Y_positive"] * 50  # 200 steps

    elif terrain_name == "random":
        #TerrainTypeList = ["flat", "X_positive", "X_negative", "Y_positive", "Y_negative"]
        # 4 types of terrain, no flat
        TerrainTypeList = ["X_positive",
                           "X_negative", "Y_positive", "Y_negative"]
        terrain_pattern = np.random.choice(TerrainTypeList, 200)
    else:
        raise Exception("Unknown Terrain Type")

    # -----------------------------------
    #   Create container for Patches
    ContactSurfsVertice = []
    ContactSurfsTypes = []
    ContactSurfsNames = []
    AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    # ------------------------------------
    # Generate a sequence of flat patches
    for surfNum in range(2, TotalNumSurfs):
        print("Put Contact Patch ", surfNum)

        # Decide Coordinate of reference point
        if surfNum % 2 == 0:  # Even number of steps (Swing the Left)
            # ----        #p2---------------------p1
            #             |                      |
            # Sn-2(Sl0) # |         Sn           |
            # (L) #       |        (L)           |
            # ----      #p3 (ref_x, ref_y)-------p4
            #             |                      |
            # Sn-1(Sr0) # |                      |
            # (R) #       |                      |

            # Define which Contact foot is for the Surface
            SurfContactLeft = "left"
            # ref_x decided by the P1_x of Sn-2 (-2 in index)
            ref_x = AllPatches[-2][0][0]
            # ref_y decided by the P1_y of Sn-1 (-1 in index)
            ref_y = AllPatches[-1][0][1]
            #center = getCenter(AllPatches[-1])
            # center[2] #ref_z decided by the central z of Sn-1 (relative foot height constraints is enforced with respect to stationary foot)
            ref_z = 0.0

        elif surfNum % 2 == 1:  # Odd number of steps
            # ----       #        ---------------------
            #             |                      |
            #             |         Sn-1         |
            #             |          (L)         |
            # ---- #      ---------------------
            #             P2 (ref_x, ref_y)------P1
            #             |                      |
            # Sn-2(Sr0) # |         Sn           |
            # (R) #       |        (R)           |
            #             P3---------------------P4

            # Define which Contact foot is for the Surface
            SurfContactLeft = "right"
            ref_x = AllPatches[-2][0][0]  # ref_x decided by the P1_x of Sn-2
            # ref_y decided by the P3_y of Sn-1 (ContactSurfsVertice[-1])
            ref_y = AllPatches[-1][2][1]
            #center = getCenter(AllPatches[-1])
            # center[2] #ref_z decided by the central z of Sn-1 (relative foot height constraints is enforced with respect to stationary foot)
            ref_z = 0.0

        # Shift along x-direction for mis alignment of First Two Patches
        # Convert from Contact Patch Sequence index (start from 0) to all patches index (+ 2)
        if surfNum == MisAlignedPatchIdx + 2:
            if randomMisAlignmentofFirstTwoPatches == True:  # if we randomly mis align the first two patches
                # MisAlignmentAmount = np.random.uniform(0.0,MaxMisAllignmentAmount) #random choose a mis aligned amount
                ref_x = ref_x + MisAlignmentAmount  # shift the x-ref of the selected patch

        #Give gaps to the patches
        if gap_between_patches == True:
            if surfNum > 2: #we give gap starting from the 4th patch (right column)
                ref_x = ref_x + x_gap

        # Build an initial flat surface
        surf_temp = flat_patch_gen(PatchColumn=SurfContactLeft, ref_x=ref_x,
                                   ref_y=ref_y, ref_z=ref_z, Proj_Length=Proj_Length, Proj_Width=Proj_Width)

        # Add to current surface to lists
        ContactSurfsVertice.append(surf_temp)
        ContactSurfsNames.append("S"+str(surfNum-2))
        ContactSurfsTypes.append(terrain_pattern[surfNum])
        AllPatches.append(surf_temp)

    # -----------------------------------
    # Modify the patches
    # Clear All Patches (because it finishes its duty)
    AllPatches = []
    # ----------------------------------------
    # (Randomly) Shrink size if we want (only make smaller)
    if random_surfsize == True:
        for ContactSurfNum in range(len(ContactSurfsVertice)):
            shrinkFactor = np.random.uniform(
                min_shrink_factor, max_shrink_factor)
            ContactSurfsVertice[ContactSurfNum] = flat_patch_shrink(
                surf=ContactSurfsVertice[ContactSurfNum], shrinkFactor=shrinkFactor, Proj_Length=Proj_Length, Proj_Width=Proj_Width)
            print("Shrink (Smaller) Contact Factor of Patch ",
                  ContactSurfNum, "with Factor of ", shrinkFactor)

    # Update AllPatches with newly genreated ContactSurfVertices
    AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    # -----------------------------------------------------------
    # Move the patches (in X Y directions only) (NOTE: We need to ensure no over shoot)
    if randomHorizontalMove == True:
        MovingDirectionList = ["X_positive",
                               "X_negative", "Y_positive", "Y_negative"]
        for surfNum in range(2, TotalNumSurfs):
            # Decide Coordinate of reference point
            if surfNum % 2 == 0:  # Even number of steps (Swing the Left)
                SurfContactLeft = "left"
                movingDirection = np.random.choice(MovingDirectionList, 1)
                if movingDirection == "Y_positive" or movingDirection == "X_positive":  # Moving outwards need to be controlled
                    movePortion = np.random.uniform(0.0, 0.1)
                else:
                    movePortion = np.random.uniform(0.0, 1.0)
            elif surfNum % 2 == 1:  # Odd number of steps (Swing the Right)
                SurfContactLeft = "right"
                if movingDirection == "Y_negative" or movingDirection == "X_positive":  # Moving outwards need to be controlled
                    movePortion = np.random.uniform(0.0, 0.1)
                else:
                    movePortion = np.random.uniform(0.0, 1.0)

            # Move Patches
            movedSurf = patch_move_horizontal_percentage(surf=AllPatches[surfNum], PatchColumn=SurfContactLeft, SurfIndex=surfNum,
                                                         Direction=movingDirection, PercentageofMovingMargin=movePortion, AllSurfaces=AllPatches)
            AllPatches[surfNum] = movedSurf
            ContactSurfsVertice[surfNum-2] = movedSurf
            print("Move Contact Patch ", surfNum-2, "Horizontally along Direction of ",
                  movingDirection, "with portion of ", movePortion)

    # -----------------------------------------
    # Rotate the patches (can change rotation axis)
    # Clear All Patches (because it finishes its duty)
    AllPatches = []
    for ContactSurfNum in range(len(ContactSurfsVertice)):
        #If we enable the large slope to exist
        if large_slope_flag == True:
            # For Normal Patches
            if not (ContactSurfNum in large_slope_index):
                current_direction = terrain_pattern[ContactSurfNum]
                current_inclination = fixed_inclination
            # For Large Slope Patches
            elif (ContactSurfNum in large_slope_index):
                # Get index in large_slope arrays
                idx_in_large_slope_array = large_slope_index.index(ContactSurfNum)
                current_direction = large_slope_directions[idx_in_large_slope_array]
                current_inclination = large_slope_inclinations[idx_in_large_slope_array]
        #No large slope enabled
        elif large_slope_flag == False:
            current_direction = terrain_pattern[ContactSurfNum]
            current_inclination = fixed_inclination

        #rotate patches
        rotatedPatch = rotate_patch(
            surf=ContactSurfsVertice[ContactSurfNum], PatchType=current_direction, theta=current_inclination)
        ContactSurfsVertice[ContactSurfNum] = rotatedPatch

        print("Rotate Patch ", ContactSurfNum, " along the direction of ", current_direction,
              "with inclination of ", current_inclination, "(None means Random)")

    # Update AllPatches with newly genreated ContactSurfVertices
    AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    # -------------------------------------------
    # Patch Elevation Changes (include non-change)
    for surfNum in range(2, TotalNumSurfs):
        # Compute elevation shift
        # For normal Patches
        if not ((surfNum-2) in large_slope_index):
            if randomElevationShift == True:
                elevation_dist = np.random.uniform(
                    min_elevation_shift, max_elevation_shift)
            else:
                elevation_dist = 0.0
        # For Large Slopes
        elif ((surfNum-2) in large_slope_index):
            idx_in_large_slope_array = large_slope_index.index(surfNum-2)
            elevation_dist = large_slope_Z_shifts[idx_in_large_slope_array]

        # manipulate the patch
        elevated_patch = elevation_shift_wrt_adjacent_patch(
            current_patch=AllPatches[surfNum], previous_patch=AllPatches[-1], elevation_shift=elevation_dist)

        print("Elevation change of Patch ", surfNum-2, "with ",
              elevation_dist, "with respect to the previous patch")

        # Rebuild the array
        AllPatches[surfNum] = elevated_patch
        ContactSurfsVertice[surfNum-2] = elevated_patch

    # ------------
    # Build Useful arrays (Only for Contact Surfaces, NOTE the index)
    #   Build containers
    ContactSurfsHalfSpace = []
    ContactSurfsTypes = []
    ContactSurfsTangentX = []
    ContactSurfsTangentY = []
    ContactSurfsNorm = []
    ContactSurfsOrientation = []

    #   Get useful arrays
    for ContactSurfNum in range(len(ContactSurfsVertice)):
        HalfSpace_temp = np.concatenate(convert_surface_to_inequality(
            ContactSurfsVertice[ContactSurfNum].T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            ContactSurfsVertice[ContactSurfNum])

        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsTypes.append(terrain_pattern[ContactSurfNum])
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

    # Build All Patches array
    AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    # ------------
    # Build Terrain Model Vector
    TerrainModel = {"InitLeftSurfVertice": Sl0,  "InitLeftSurfType": LeftSurfType,
                    "InitLeftSurfTangentX": Sl0_TangentX, "InitLeftSurfTangentY": Sl0_TangentY, "InitLeftSurfNorm": Sl0_Norm, "InitLeftSurfOrientation": Sl0_Orientation,
                    "InitRightSurfVertice": Sr0, "InitRightSurfType": RightSurfType,
                    "InitRightSurfTangentX": Sr0_TangentX, "InitRightSurfTangentY": Sr0_TangentY, "InitRightSurfNorm": Sr0_Norm, "InitRightSurfOrientation": Sr0_Orientation,
                    "ContactSurfsVertice": ContactSurfsVertice,
                    "ContactSurfsHalfSpace": ContactSurfsHalfSpace,
                    "ContactSurfsTypes": ContactSurfsTypes,
                    "ContactSurfsNames": ContactSurfsNames,
                    "ContactSurfsTangentX": ContactSurfsTangentX,
                    "ContactSurfsTangentY": ContactSurfsTangentY,
                    "ContactSurfsNorm": ContactSurfsNorm,
                    "ContactSurfsOrientation": ContactSurfsOrientation,
                    "AllPatchesVertices": AllPatches}

    return TerrainModel

# given a reference point, we expand to 4 vertices of patches


def flat_patch_gen(PatchColumn="left", ref_x=0, ref_y=0, ref_z=0, Proj_Length=0.6, Proj_Width=0.6):

    # Calculate Projected Vertices
    # For Left Patch
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4
    # ref_x,ref_y,ref_z
    if PatchColumn == "left":
        Proj_p1 = [ref_x + Proj_Length, ref_y + Proj_Width, ref_z]
        Proj_p2 = [ref_x,               ref_y + Proj_Width, ref_z]
        Proj_p3 = [ref_x,               ref_y,              ref_z]
        Proj_p4 = [ref_x + Proj_Length, ref_y,              ref_z]

    # For Right Patch
    #ref_x, ref_y
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4
    elif PatchColumn == "right":
        Proj_p1 = [ref_x + Proj_Length, ref_y,                           ref_z]
        Proj_p2 = [ref_x,               ref_y,                           ref_z]
        Proj_p3 = [ref_x,               ref_y - Proj_Width,              ref_z]
        Proj_p4 = [ref_x + Proj_Length, ref_y - Proj_Width,              ref_z]

    # Copy projected patch to output patch
    p1 = Proj_p1
    p2 = Proj_p2
    p3 = Proj_p3
    p4 = Proj_p4

    # Build Surface array
    Surface = np.array([p1, p2, p3, p4])

    return Surface

# shrink the patch size (only make smaller)


def flat_patch_shrink(surf=None, shrinkFactor=1.0, Proj_Length=None, Proj_Width=None):
    # patch vertices
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4

    length_shrink = shrinkFactor*Proj_Length
    width_shrink = shrinkFactor*Proj_Width

    shrinksurf = surf + np.array([[-length_shrink/2.0, -width_shrink/2.0, 0.0],
                                  [length_shrink/2.0, -width_shrink/2.0, 0.0],
                                  [length_shrink/2.0,  width_shrink/2.0, 0.0],
                                  [-length_shrink/2.0,  width_shrink/2.0, 0.0]])

    return shrinksurf

# Move flat patches horizontally while consider not overshooting the adjacent up/down, left and right patches
# moving portion is defined as percentage of the margin to the left/right, up/down patc borders


def patch_move_horizontal_percentage(surf=None, PatchColumn="left", SurfIndex=None, Direction=None, PercentageofMovingMargin=None, AllSurfaces=None):
    # patch vertices
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4

    if PatchColumn == "left":
        # Left  Column |  Patch(n-2)   |   *Patch(n)*    |  Patch(n+2)   |
        # Right Column |  patch(n-1)   |   Patch(n+1)    |  Patch(n+3)   |
        if Direction == "X_positive":

            # NOTE: len(AllSurfaces) - 1 is the index of the last patch, if SurfIndex + 2 is larger than this value, then we reach the right most border of the terrain
            if SurfIndex + 2 > len(AllSurfaces) - 1:
                # margin = 1/2patch length: [current][p1][x] - [current][p2][x]
                margin = np.abs(
                    AllSurfaces[SurfIndex][0][0] - AllSurfaces[SurfIndex][1][0])
            else:
                #[current + 2][p2][x] - [current][p1][x]
                margin = np.abs(
                    AllSurfaces[SurfIndex + 2][1][0] - AllSurfaces[SurfIndex][0][0])
            # Compute moving distance
            movedist = margin*PercentageofMovingMargin
            movesurf = surf + np.array([[movedist, 0.0, 0.0],
                                        [movedist, 0.0, 0.0],
                                        [movedist, 0.0, 0.0],
                                        [movedist, 0.0, 0.0]])

        elif Direction == "X_negative":
            #margin = [current][p2][x] - [current - 2][p1][x]
            margin = np.abs(AllSurfaces[SurfIndex][1]
                            [0] - AllSurfaces[SurfIndex - 2][0][0])
            # Compute moving distance
            movedist = margin*PercentageofMovingMargin
            movesurf = surf + np.array([[-movedist, 0.0, 0.0],
                                        [-movedist, 0.0, 0.0],
                                        [-movedist, 0.0, 0.0],
                                        [-movedist, 0.0, 0.0]])
        elif Direction == "Y_positive":
            # margin = 1/2 current patch width: [current][p1][y] - [current][p4][y]
            margin = np.abs(AllSurfaces[SurfIndex]
                            [0][1] - AllSurfaces[SurfIndex][3][1])
            # Compute moving distance
            movedist = margin*PercentageofMovingMargin
            movesurf = surf + np.array([[0.0, movedist, 0.0],
                                        [0.0, movedist, 0.0],
                                        [0.0, movedist, 0.0],
                                        [0.0, movedist, 0.0]])
        elif Direction == "Y_negative":
            # NOTE: len(AllSurfaces) - 1 is the index of the last patch, if SurfIndex + 1 is larger than this value, then we reach the right most border of the terrain
            if SurfIndex + 1 > len(AllSurfaces) - 1:
                # margin = 1/2 patch width: [current][p1][y] - [current][p4][y]
                margin = np.abs(
                    AllSurfaces[SurfIndex][0][1] - AllSurfaces[SurfIndex][3][1])
            else:
                #margin = [current][p3][y] - [current + 1][p1][x]
                margin = np.abs(
                    AllSurfaces[SurfIndex][2][1] - AllSurfaces[SurfIndex + 1][0][1])

            # Compute moving distance
            movedist = margin*PercentageofMovingMargin
            movesurf = surf + np.array([[0.0, -movedist, 0.0],
                                        [0.0, -movedist, 0.0],
                                        [0.0, -movedist, 0.0],
                                        [0.0, -movedist, 0.0]])

    elif PatchColumn == "right":
        # Left  Column |  Patch(n-3)   |   Patch(n-1)    |  Patch(n+1)   |
        # Right Column |  patch(n-2)   |   *Patch(n)*    |  Patch(n+2)   |

        # [current + 2][p2][x] - [current][p1][x]
        if Direction == "X_positive":
            # NOTE: len(AllSurfaces) - 1 is the index of the last patch, if SurfIndex + 2 is larger than this value, then we reach the right most border of the terrain
            if SurfIndex + 2 > len(AllSurfaces) - 1:
                # margin = 1/2patch length: [current][p1][x] - [current][p2][x]
                margin = np.abs(
                    AllSurfaces[SurfIndex][0][0] - AllSurfaces[SurfIndex][1][0])
            else:
                #margin = [current + 2][p2][x] - [current][p1][x]
                margin = np.abs(
                    AllSurfaces[SurfIndex + 2][1][0] - AllSurfaces[SurfIndex][0][0])
            movedist = margin*PercentageofMovingMargin
            movesurf = surf + np.array([[movedist, 0.0, 0.0],
                                        [movedist, 0.0, 0.0],
                                        [movedist, 0.0, 0.0],
                                        [movedist, 0.0, 0.0]])
        elif Direction == "X_negative":
            #margin = [current][p2][x] - [current - 2][p1][x]
            margin = np.abs(AllSurfaces[SurfIndex][1]
                            [0] - AllSurfaces[SurfIndex - 2][0][0])
            movedist = margin*PercentageofMovingMargin
            movesurf = surf + np.array([[-movedist, 0.0, 0.0],
                                        [-movedist, 0.0, 0.0],
                                        [-movedist, 0.0, 0.0],
                                        [-movedist, 0.0, 0.0]])
        elif Direction == "Y_positive":
            #margin = [current - 1][p3][y] - [current][p1][y]
            margin = np.abs(AllSurfaces[SurfIndex - 1]
                            [2][1] - AllSurfaces[SurfIndex][0][1])
            movedist = margin*PercentageofMovingMargin
            movesurf = surf + np.array([[0.0, movedist, 0.0],
                                        [0.0, movedist, 0.0],
                                        [0.0, movedist, 0.0],
                                        [0.0, movedist, 0.0]])
        elif Direction == "Y_negative":
            # margin = 1/2 current patch width: [current][p1][y] - [current][p4][y]
            margin = np.abs(AllSurfaces[SurfIndex]
                            [0][1] - AllSurfaces[SurfIndex][3][1])
            movedist = margin*PercentageofMovingMargin
            movesurf = surf + np.array([[0.0, -movedist, 0.0],
                                        [0.0, -movedist, 0.0],
                                        [0.0, -movedist, 0.0],
                                        [0.0, -movedist, 0.0]])

    return movesurf


def patch_move_XYZ_distance(surf=None, PatchColumn="left", X_shift=0.0, Y_shift=0.0, Z_shift=0.0):

    # Get the rotated patch
    shiftedSurf = surf

    # Shift along X-Direction
    shiftedSurf = shiftedSurf + np.array([[X_shift, 0.0, 0.0],
                                          [X_shift, 0.0, 0.0],
                                          [X_shift, 0.0, 0.0],
                                          [X_shift, 0.0, 0.0]])

    # Shift along Y-Direction (NOTE:Positive means Outward, away from x=0, Negative means moving inward, towards x=0, different for left and right patches)
    if PatchColumn == "left":
        shiftedSurf = shiftedSurf + np.array([[0.0, Y_shift, 0.0],
                                              [0.0, Y_shift, 0.0],
                                              [0.0, Y_shift, 0.0],
                                              [0.0, Y_shift, 0.0]])
    elif PatchColumn == "right":
        shiftedSurf = shiftedSurf + np.array([[0.0, -Y_shift, 0.0],
                                              [0.0, -Y_shift, 0.0],
                                              [0.0, -Y_shift, 0.0],
                                              [0.0, -Y_shift, 0.0]])
    else:
        raise Exception("Unknown Patch Column")

    # Shift along Z-Direction
    shiftedSurf = shiftedSurf + np.array([[0.0, 0.0, Z_shift],
                                          [0.0, 0.0, Z_shift],
                                          [0.0, 0.0, Z_shift],
                                          [0.0, 0.0, Z_shift]])

    return shiftedSurf

# max_theta slightly smaller than the friction cone (0.3)


def rotate_patch(surf=None, PatchType=None, theta=None, min_theta=0.08, max_theta=0.2):
    print(PatchType)
    
    # get the flat patch
    rotatedPatch = surf

    # randomly Sample a rotation angle (in case of rotation patch), ip theta is not provided
    if theta == None:
        # round to the third decimal
        theta = np.round(np.random.uniform(min_theta, max_theta), 3)

    # Calculate (current) Patch Length and Width
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4
    # Patch Length = abs([p1][x] - [p2][x])
    patchLength = np.abs(surf[0][0] - surf[1][0])
    # Patch Width = abs([p1][y] - [p4][y])
    patchWidth = np.abs(surf[0][1] - surf[3][1])

    if PatchType == "flat":
        print("TerrainGen: flat patch - no changes to projected patch")
    elif PatchType == "X_positive":
        # theta = rotation angle, defined in radius
        #theta = np.arctan2(0.3,1)
        print("TerrainGen Rotation along X positive, with ",
              str(theta/np.pi*180), "degrees")
        delta_z = patchWidth/2*np.tan(theta)
        # p2---------------------p1
        # |                      |
        # |                      |   ------>x
        # |                      |
        # p3---------------------p4
        rotatedPatch[0][2] = rotatedPatch[0][2] + delta_z
        rotatedPatch[1][2] = rotatedPatch[1][2] + delta_z
        rotatedPatch[2][2] = rotatedPatch[2][2] - delta_z
        rotatedPatch[3][2] = rotatedPatch[3][2] - delta_z

    elif PatchType == "X_negative":
        # theta = rotation angle, defined in radius
        #theta = np.arctan2(0.3,1)
        print("TerrainGen Rotation along X negative, with ",
              str(theta/np.pi*180), "degrees")
        delta_z = patchWidth/2*np.tan(theta)
        # p2---------------------p1
        # |                      |
        # |                      |   ------>x
        # |                      |
        # p3---------------------p4
        rotatedPatch[0][2] = rotatedPatch[0][2] - delta_z
        rotatedPatch[1][2] = rotatedPatch[1][2] - delta_z
        rotatedPatch[2][2] = rotatedPatch[2][2] + delta_z
        rotatedPatch[3][2] = rotatedPatch[3][2] + delta_z

    elif PatchType == "Y_positive":
        # theta = rotation angle, defined in radius
        #theta = np.arctan2(0.3,1)
        print("TerrainGen Rotation along Y positive, with ",
              str(theta/np.pi*180), "degrees")
        delta_z = patchLength/2.0*np.tan(theta)
        # p2---------------------p1
        # |                      |
        # |                      |
        # |                      |
        # p3---------------------p4
        #           ^y
        rotatedPatch[0][2] = rotatedPatch[0][2] - delta_z
        rotatedPatch[1][2] = rotatedPatch[1][2] + delta_z
        rotatedPatch[2][2] = rotatedPatch[2][2] + delta_z
        rotatedPatch[3][2] = rotatedPatch[3][2] - delta_z
    elif PatchType == "Y_negative":
        #theta = np.arctan2(0.3,1)
        print("TerrainGen Rotation along Y negative, with ",
              str(theta/np.pi*180), "degrees")
        delta_z = patchLength/2.0*np.tan(theta)
        # p2---------------------p1
        # |                      |
        # |                      |
        # |                      |
        # p3---------------------p4
        #           ^y
        rotatedPatch[0][2] = rotatedPatch[0][2] + delta_z
        rotatedPatch[1][2] = rotatedPatch[1][2] - delta_z
        rotatedPatch[2][2] = rotatedPatch[2][2] - delta_z
        rotatedPatch[3][2] = rotatedPatch[3][2] + delta_z

    elif PatchType == "DiagX_positive":
        # p2---------------------p1 (*on rotation axis, positive direction)
        # |                 -     |
        # |         -             |
        # |  -                    |
        # p3---------------------p4
        # (on rotation axis)
        print("TerrainGen Rotation along Diagonal X postive, with ",
              str(theta/np.pi*180), "degrees")
        
        proj_diagonal_length = np.sqrt(patchLength**2 + patchWidth**2)
        delta_z = proj_diagonal_length/2.0*np.tan(theta)
        
        rotatedPatch[0][2] = rotatedPatch[0][2]
        rotatedPatch[1][2] = rotatedPatch[1][2] + delta_z
        rotatedPatch[2][2] = rotatedPatch[2][2]
        rotatedPatch[3][2] = rotatedPatch[3][2] - delta_z

    elif PatchType == "DiagX_negative":
        #use Projected diagonal length, because 

        # p2---------------------p1 (on rotation axis)
        # |                 -     |
        # |         -             |
        # |  -                    |
        # p3---------------------p4
        # (*on rotation axis, on positive direction)
        print("TerrainGen Rotation along Diagonal X negative, with ",
              str(theta/np.pi*180), "degrees")
        
        proj_diagonal_length = np.sqrt(patchLength**2 + patchWidth**2)
        delta_z = proj_diagonal_length/2.0*np.tan(theta)
        
        rotatedPatch[0][2] = rotatedPatch[0][2]
        rotatedPatch[1][2] = rotatedPatch[1][2] - delta_z
        rotatedPatch[2][2] = rotatedPatch[2][2]
        rotatedPatch[3][2] = rotatedPatch[3][2] + delta_z

    elif PatchType == "DiagY_positive":
        #(*on rotation axis, positive direction)
        # p2---------------------p1 
        # | -                     |
        # |         -             |
        # |                  -    |
        # p3---------------------p4
        #                           (on rotation axis)
        print("TerrainGen Rotation along Diagonal Y postive, with ",
              str(theta/np.pi*180), "degrees")
        
        proj_diagonal_length = np.sqrt(patchLength**2 + patchWidth**2)
        delta_z = proj_diagonal_length/2.0*np.tan(theta)
        
        rotatedPatch[0][2] = rotatedPatch[0][2] - delta_z
        rotatedPatch[1][2] = rotatedPatch[1][2] 
        rotatedPatch[2][2] = rotatedPatch[2][2] + delta_z
        rotatedPatch[3][2] = rotatedPatch[3][2] 

    elif PatchType == "DiagY_negative":

        #(on rotation axis)
        # p2---------------------p1 
        # | -                     |
        # |         -             |
        # |                  -    |
        # p3---------------------p4
        #                           (*on rotation axis, positive direction)
        print("TerrainGen Rotation along Diagonal Y negative, with ",
              str(theta/np.pi*180), "degrees")
        
        proj_diagonal_length = np.sqrt(patchLength**2 + patchWidth**2)
        delta_z = proj_diagonal_length/2.0*np.tan(theta)
        
        rotatedPatch[0][2] = rotatedPatch[0][2] + delta_z
        rotatedPatch[1][2] = rotatedPatch[1][2] 
        rotatedPatch[2][2] = rotatedPatch[2][2] - delta_z
        rotatedPatch[3][2] = rotatedPatch[3][2] 

    else:
        raise Exception("Unknown Patch Type")

    return rotatedPatch


def pre_designed_terrain(terrain_name="flat", NumSteps=None, NumLookAhead=None, LargeSlopeAngle=None):
    # Surf Vertex Identification
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4

    # For contact surfaces only (not included intial two contact patches)
    TotalNumSurfs = NumSteps + NumLookAhead - 1

    #   Make a containers
    ContactSurfsVertice = []
    ContactSurfsHalfSpace = []
    ContactSurfsNames = []
    ContactSurfsTypes = []
    ContactSurfsTangentX = []
    ContactSurfsTangentY = []
    ContactSurfsNorm = []
    ContactSurfsOrientation = []

    # if terrain_name == "single_stair":
    #Patch1 = np.array([[0.4, 0.5, 0.], [-0.1, 0.5, 0.], [-0.1, -0.5, 0.], [0.4, -0.5, 0.]])
    #Patch2 = np.array([[0.7, 0.5, 0.1], [0.4, 0.5, 0.1], [0.4, -0.5, 0.1], [0.7, -0.5, 0.1]])
    #Patch3 = np.array([[5, 0.5, 0.], [0.7, 0.5, 0.], [0.7, -0.5, 0.], [5, -0.5, 0.]])
    #AllPatches = [Patch1,Patch2,Patch3]
    if terrain_name == "single_large_slope_far":
        # Define Initial Patches
        print("Left Init Contact Surface:")
        # initial patch for the left foot (for the first step only)
        Sl0 = np.array([[1.5, 1, 0.], [-1, 1, 0.],
                       [-1, -1, 0.], [1.5, -1, 0.]])
        LeftSurfType = getSurfaceType(Sl0)  # "flat"
        print("Left Init Contact Surface Type: ", LeftSurfType)
        Sl0_TangentX, Sl0_TangentY, Sl0_Norm, Sl0_Orientation = getTerrainTagentsNormOrientation(
            Sl0)

        print("Right Init Contact Surface")
        # initial patch for the right foot (for the first step only)
        Sr0 = np.array([[1.5, 1, 0.], [-1, 1, 0.],
                       [-1, -1, 0.], [1.5, -1, 0.]])
        RightSurfType = getSurfaceType(Sr0)  # "flat"
        print("Right Init Contact Surface Type: ", RightSurfType)
        Sr0_TangentX, Sr0_TangentY, Sr0_Norm, Sr0_Orientation = getTerrainTagentsNormOrientation(
            Sr0)

        # Add a Flat
        Patch_temp = np.array(
            [[1.5, 1, 0.], [-1, 1, 0.], [-1, -1, 0.], [1.5, -1, 0.]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S0")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        # Add a Flat
        Patch_temp = np.array(
            [[1.5, 1, 0.], [-1, 1, 0.], [-1, -1, 0.], [1.5, -1, 0.]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S1")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        # Add a Flat
        Patch_temp = np.array(
            [[1.5, 1, 0.], [-1, 1, 0.], [-1, -1, 0.], [1.5, -1, 0.]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S2")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        # Add a Flat
        Patch_temp = np.array(
            [[1.5, 1, 0.], [-1, 1, 0.], [-1, -1, 0.], [1.5, -1, 0.]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S3")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        # Three Flats in Total

        # Add the Slope
        slope = 0.5
        SlopePatch_Vertice = np.array(
            [[1.8, 1, slope-0.02], [1.5, 1, slope-0.02], [1.5, -1.0, -slope-0.02], [1.8, -1.0, -slope-0.02]])
        Slope_HalfSpace = np.concatenate(convert_surface_to_inequality(
            SlopePatch_Vertice.T), axis=None)  # output tuple, we make into an array
        SlopeTangentX, SlopeTangentY, SlopeNorm, SlopeOrientation = getTerrainTagentsNormOrientation(
            SlopePatch_Vertice)
        ContactSurfsVertice.append(SlopePatch_Vertice)
        ContactSurfsHalfSpace.append(Slope_HalfSpace)
        ContactSurfsNames.append("S4")
        ContactSurfsTypes.append(getSurfaceType(SlopePatch_Vertice))
        ContactSurfsTangentX.append(SlopeTangentX)
        ContactSurfsTangentY.append(SlopeTangentY)
        ContactSurfsNorm.append(SlopeNorm)
        ContactSurfsOrientation.append(SlopeOrientation)

        # Add the Flat Patch
        if TotalNumSurfs > 5:
            for surfcnt in range(TotalNumSurfs-5):
                Patch_temp = np.array(
                    [[10, 1, 0.], [1.8, 1, 0.], [1.8, -1, 0.], [10, -1, 0.]])
                HalfSpace_temp = np.concatenate(
                    convert_surface_to_inequality(Patch_temp.T), axis=None)
                TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
                    Patch_temp)
                ContactSurfsVertice.append(Patch_temp)
                ContactSurfsHalfSpace.append(HalfSpace_temp)
                ContactSurfsNames.append("S"+str(surfcnt+5))
                ContactSurfsTypes.append(getSurfaceType(Patch_temp))
                ContactSurfsTangentX.append(TangentX_temp)
                ContactSurfsTangentY.append(TangentY_temp)
                ContactSurfsNorm.append(Norm_temp)
                ContactSurfsOrientation.append(OrientationTemp)

        AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    elif terrain_name == "single_large_slope_close":
        # Define Initial Patches
        print("Left Init Contact Surface:")
        # initial patch for the left foot (for the first step only)
        Sl0 = np.array([[0.3, 1, 0.], [-1, 1, 0.],
                       [-1, -1, 0.], [0.3, -1, 0.]])
        LeftSurfType = getSurfaceType(Sl0)  # "flat"
        print("Left Init Contact Surface Type: ", LeftSurfType)
        Sl0_TangentX, Sl0_TangentY, Sl0_Norm, Sl0_Orientation = getTerrainTagentsNormOrientation(
            Sl0)

        print("Right Init Contact Surface")
        # initial patch for the right foot (for the first step only)
        Sr0 = np.array([[0.3, 1, 0.], [-1, 1, 0.],
                       [-1, -1, 0.], [0.3, -1, 0.]])
        RightSurfType = getSurfaceType(Sr0)  # "flat"
        print("Right Init Contact Surface Type: ", RightSurfType)
        Sr0_TangentX, Sr0_TangentY, Sr0_Norm, Sr0_Orientation = getTerrainTagentsNormOrientation(
            Sr0)

        # Three Flats in Total

        # Add the Slope
        SlopePatch_Vertice = np.array(
            [[0.5, 1, 0.4], [0.3, 1, 0.4], [0.3, -1, -0.4], [0.5, -1, -0.4]])
        Slope_HalfSpace = np.concatenate(convert_surface_to_inequality(
            SlopePatch_Vertice.T), axis=None)  # output tuple, we make into an array
        SlopeTangentX, SlopeTangentY, SlopeNorm, SlopeOrientation = getTerrainTagentsNormOrientation(
            SlopePatch_Vertice)
        ContactSurfsVertice.append(SlopePatch_Vertice)
        ContactSurfsHalfSpace.append(Slope_HalfSpace)
        ContactSurfsNames.append("S0")
        ContactSurfsTypes.append(getSurfaceType(SlopePatch_Vertice))
        ContactSurfsTangentX.append(SlopeTangentX)
        ContactSurfsTangentY.append(SlopeTangentY)
        ContactSurfsNorm.append(SlopeNorm)
        ContactSurfsOrientation.append(SlopeOrientation)

        # Add the Flat Patch
        if TotalNumSurfs > 1:
            for surfcnt in range(TotalNumSurfs-1):
                Patch_temp = np.array(
                    [[10, 1, 0.], [0.5, 1, 0.], [0.5, -1, 0.], [10, -1, 0.]])
                HalfSpace_temp = np.concatenate(
                    convert_surface_to_inequality(Patch_temp.T), axis=None)
                TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
                    Patch_temp)
                ContactSurfsVertice.append(Patch_temp)
                ContactSurfsHalfSpace.append(HalfSpace_temp)
                ContactSurfsNames.append("S"+str(surfcnt+1))
                ContactSurfsTypes.append(getSurfaceType(Patch_temp))
                ContactSurfsTangentX.append(TangentX_temp)
                ContactSurfsTangentY.append(TangentY_temp)
                ContactSurfsNorm.append(Norm_temp)
                ContactSurfsOrientation.append(OrientationTemp)

        AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    elif terrain_name == "darpa_left":

        startx = 0.2

        print("Left Init Contact Surface:")
        Sl0 = np.array([[startx, 0.6, 0.], [-0.2, 0.6, 0.],
                       [-0.2, 0.0, 0.], [startx, 0.0, 0.0]])
        LeftSurfType = getSurfaceType(Sl0)  # "flat"
        print("Left Init Contact Surface Type: ", LeftSurfType)
        Sl0_TangentX, Sl0_TangentY, Sl0_Norm, Sl0_Orientation = getTerrainTagentsNormOrientation(
            Sl0)

        print("Right Init Contact Surface")
        Sr0 = np.array([[startx, 0, 0.], [-0.2, 0, 0.],
                       [-0.2, -0.6, 0.], [startx, -0.6, 0.0]])
        RightSurfType = getSurfaceType(Sr0)  # "flat"
        print("Right Init Contact Surface Type: ", RightSurfType)
        Sr0_TangentX, Sr0_TangentY, Sr0_Norm, Sr0_Orientation = getTerrainTagentsNormOrientation(
            Sr0)

        HightVariation = 0.05
        HorizontanIncrease = 0.6

        Patch_temp = np.array([[startx+HorizontanIncrease, 0.6, -HightVariation], [startx, 0.6, HightVariation], [
                              startx, 0, HightVariation], [startx+HorizontanIncrease, 0, -HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S0")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+HorizontanIncrease, 0, -HightVariation], [startx, 0, -HightVariation], [
                              startx, -0.6, HightVariation], [startx+HorizontanIncrease, -0.6, HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S1")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+2*HorizontanIncrease, 0.6, HightVariation], [startx+HorizontanIncrease, 0.6, -HightVariation], [
                              startx+HorizontanIncrease, 0, -HightVariation], [startx+2*HorizontanIncrease, 0, HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S2")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+2*HorizontanIncrease, 0, HightVariation], [startx+HorizontanIncrease, 0, HightVariation], [
                              startx+HorizontanIncrease, -0.6, -HightVariation], [startx+2*HorizontanIncrease, -0.6, -HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S3")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        LargeSlopeHeight = 0.1
        #Patch_temp = np.array([[startx+3*HorizontanIncrease, 0.6, HightVariation], [startx+2*HorizontanIncrease, 0.6, HightVariation], [startx+2*HorizontanIncrease, 0, -HightVariation], [startx+3*HorizontanIncrease, 0, -HightVariation]])
        Patch_temp = np.array([[startx+3*HorizontanIncrease, 0.6, LargeSlopeHeight], [startx+2*HorizontanIncrease, 0.6, LargeSlopeHeight], [
                              startx+2*HorizontanIncrease, 0, -LargeSlopeHeight], [startx+3*HorizontanIncrease, 0, -LargeSlopeHeight]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S4")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+3*HorizontanIncrease, 0, -HightVariation], [startx+2*HorizontanIncrease, 0, HightVariation], [
                              startx+2*HorizontanIncrease, -0.6, HightVariation], [startx+3*HorizontanIncrease, -0.6, -HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S5")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+4*HorizontanIncrease, 0.6, HightVariation], [startx+3*HorizontanIncrease, 0.6, -HightVariation], [
                              startx+3*HorizontanIncrease, 0, -HightVariation], [startx+4*HorizontanIncrease, 0, HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S6")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+4*HorizontanIncrease, 0, HightVariation], [startx+3*HorizontanIncrease, 0, HightVariation], [
                              startx+3*HorizontanIncrease, -0.6, -HightVariation], [startx+4*HorizontanIncrease, -0.6, -HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S7")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+5*HorizontanIncrease, 0.6, -HightVariation], [startx+4*HorizontanIncrease, 0.6, HightVariation], [
                              startx+4*HorizontanIncrease, 0, HightVariation], [startx+5*HorizontanIncrease, 0, -HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S8")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+5*HorizontanIncrease, 0, -HightVariation], [startx+4*HorizontanIncrease, 0, -HightVariation], [
                              startx+4*HorizontanIncrease, -0.6, HightVariation], [startx+5*HorizontanIncrease, -0.6, HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S9")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[5.0, 0.6, 0.], [startx+5*HorizontanIncrease, 0.6, 0.], [
                              startx+5*HorizontanIncrease, -0.6, 0.], [5.0, -0.6, 0.]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S10")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        if TotalNumSurfs > 11:
            for surfcnt in range(TotalNumSurfs-11):
                Patch_temp = np.array([[5.0, 0.6, 0.], [startx+5*HorizontanIncrease, 0.6, 0.], [
                                      startx+5*HorizontanIncrease, -0.6, 0.], [5.0, -0.6, 0.]])
                HalfSpace_temp = np.concatenate(
                    convert_surface_to_inequality(Patch_temp.T), axis=None)
                TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
                    Patch_temp)
                ContactSurfsVertice.append(Patch_temp)
                ContactSurfsHalfSpace.append(HalfSpace_temp)
                ContactSurfsNames.append("S"+str(surfcnt+11))
                ContactSurfsTypes.append(getSurfaceType(Patch_temp))
                ContactSurfsTangentX.append(TangentX_temp)
                ContactSurfsTangentY.append(TangentY_temp)
                ContactSurfsNorm.append(Norm_temp)
                ContactSurfsOrientation.append(OrientationTemp)

        AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    elif terrain_name == "flat_small_patches":

        startx = 0.2

        print("Left Init Contact Surface:")
        Sl0 = np.array([[startx, 0.6, 0.], [-0.2, 0.6, 0.],
                       [-0.2, 0.0, 0.], [startx, 0.0, 0.0]])
        LeftSurfType = getSurfaceType(Sl0)  # "flat"
        print("Left Init Contact Surface Type: ", LeftSurfType)
        Sl0_TangentX, Sl0_TangentY, Sl0_Norm, Sl0_Orientation = getTerrainTagentsNormOrientation(
            Sl0)

        print("Right Init Contact Surface")
        Sr0 = np.array([[startx, 0, 0.], [-0.2, 0, 0.],
                       [-0.2, -0.6, 0.], [startx, -0.6, 0.0]])
        RightSurfType = getSurfaceType(Sr0)  # "flat"
        print("Right Init Contact Surface Type: ", RightSurfType)
        Sr0_TangentX, Sr0_TangentY, Sr0_Norm, Sr0_Orientation = getTerrainTagentsNormOrientation(
            Sr0)

        HightVariation = 0.07
        HorizontanIncrease = 0.5

        Patch_temp = np.array([[startx+HorizontanIncrease, 0.6, -HightVariation], [startx, 0.6, HightVariation], [
                              startx, 0, HightVariation], [startx+HorizontanIncrease, 0, -HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S0")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+HorizontanIncrease, 0, 0.0], [startx, 0, 0.0], [
                              startx, -0.6, 0.0], [startx+HorizontanIncrease, -0.6, 0.0]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S1")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+2*HorizontanIncrease, 0.6, -HightVariation], [startx+HorizontanIncrease, 0.6, -HightVariation], [
                              startx+HorizontanIncrease, 0, HightVariation], [startx+2*HorizontanIncrease, 0, HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S2")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+2*HorizontanIncrease, 0, -0.05], [startx+HorizontanIncrease, 0, -0.05], [
                              startx+HorizontanIncrease, -0.6, 0.05], [startx+2*HorizontanIncrease, -0.6, 0.05]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S3")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+3*HorizontanIncrease, 0.6, -0.045], [startx+2*HorizontanIncrease,
                              0.6, -0.045], [startx+2*HorizontanIncrease, 0, 0.045], [startx+3*HorizontanIncrease, 0, 0.045]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S4")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+3*HorizontanIncrease, 0, -HightVariation], [startx+2*HorizontanIncrease, 0, HightVariation], [
                              startx+2*HorizontanIncrease, -0.6, HightVariation], [startx+3*HorizontanIncrease, -0.6, -HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S5")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+4*HorizontanIncrease, 0.6, HightVariation], [startx+3*HorizontanIncrease, 0.6, -HightVariation], [
                              startx+3*HorizontanIncrease, 0, -HightVariation], [startx+4*HorizontanIncrease, 0, HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S6")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+4*HorizontanIncrease, 0, -HightVariation], [startx+3*HorizontanIncrease, 0, HightVariation], [
                              startx+3*HorizontanIncrease, -0.6, HightVariation], [startx+4*HorizontanIncrease, -0.6, -HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S7")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+5*HorizontanIncrease, 0.6, 0.0], [startx+4*HorizontanIncrease, 0.6, 0.0], [
                              startx+4*HorizontanIncrease, 0, 0.0], [startx+5*HorizontanIncrease, 0, 0.0]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S8")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[startx+5*HorizontanIncrease, 0, HightVariation], [startx+4*HorizontanIncrease, 0, HightVariation], [
                              startx+4*HorizontanIncrease, -0.6, -HightVariation], [startx+5*HorizontanIncrease, -0.6, -HightVariation]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S9")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        Patch_temp = np.array([[5.0, 0.6, 0.], [startx+5*HorizontanIncrease, 0.6, 0.], [
                              startx+5*HorizontanIncrease, -0.6, 0.], [5.0, -0.6, 0.]])
        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(Patch_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            Patch_temp)
        ContactSurfsVertice.append(Patch_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S10")
        ContactSurfsTypes.append(getSurfaceType(Patch_temp))
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

        if TotalNumSurfs > 11:
            for surfcnt in range(TotalNumSurfs-11):
                Patch_temp = np.array([[5.0, 0.6, 0.], [startx+5*HorizontanIncrease, 0.6, 0.], [
                                      startx+5*HorizontanIncrease, -0.6, 0.], [5.0, -0.6, 0.]])
                HalfSpace_temp = np.concatenate(
                    convert_surface_to_inequality(Patch_temp.T), axis=None)
                TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
                    Patch_temp)
                ContactSurfsVertice.append(Patch_temp)
                ContactSurfsHalfSpace.append(HalfSpace_temp)
                ContactSurfsNames.append("S"+str(surfcnt+11))
                ContactSurfsTypes.append(getSurfaceType(Patch_temp))
                ContactSurfsTangentX.append(TangentX_temp)
                ContactSurfsTangentY.append(TangentY_temp)
                ContactSurfsNorm.append(Norm_temp)
                ContactSurfsOrientation.append(OrientationTemp)

        AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    elif terrain_name == "flat":

        # Define Initial Patches
        print("Left Init Contact Surface:")
        # initial patch for the left foot (for the first step only)
        Sl0 = np.array([[20, 1, 0.], [-1, 1, 0.], [-1, -1, 0.], [20, -1, 0.]])
        LeftSurfType = getSurfaceType(Sl0)  # "flat"
        print("Left Init Contact Surface Type: ", LeftSurfType)
        Sl0_TangentX, Sl0_TangentY, Sl0_Norm, Sl0_Orientation = getTerrainTagentsNormOrientation(
            Sl0)

        print("Right Init Contact Surface")
        # initial patch for the right foot (for the first step only)
        Sr0 = np.array([[20, 1, 0.], [-1, 1, 0.], [-1, -1, 0.], [20, -1, 0.]])
        RightSurfType = getSurfaceType(Sr0)  # "flat"
        print("Right Init Contact Surface Type: ", RightSurfType)
        Sr0_TangentX, Sr0_TangentY, Sr0_Norm, Sr0_Orientation = getTerrainTagentsNormOrientation(
            Sr0)

        FlatPatch_Vertice = np.array(
            [[20, 1, 0.], [-1, 1, 0.], [-1, -1, 0.], [20, -1, 0.]])
        FlatPatch_HalfSpace = np.concatenate(convert_surface_to_inequality(
            FlatPatch_Vertice.T), axis=None)  # output tuple, we make into an array
        FlatTangentX, FlatTangentY, FlatNorm, FlatOrientation = getTerrainTagentsNormOrientation(
            FlatPatch_Vertice)

        # build Contact Paches

        #   Start Making the list
        for SurfNum in range(TotalNumSurfs):
            ContactSurfsVertice.append(FlatPatch_Vertice)
            ContactSurfsHalfSpace.append(FlatPatch_HalfSpace)
            ContactSurfsNames.append("S"+str(SurfNum))
            ContactSurfsTypes.append(getSurfaceType(FlatPatch_Vertice))
            ContactSurfsTangentX.append(FlatTangentX)
            ContactSurfsTangentY.append(FlatTangentY)
            ContactSurfsNorm.append(FlatNorm)
            ContactSurfsOrientation.append(FlatOrientation)

        AllPatches = [Sl0, Sr0] + ContactSurfsVertice

    else:
        raise Exception("Unknown Terrain Type")

    # Build Terrain Model Vector
    TerrainModel = {"InitLeftSurfVertice": Sl0,  "InitLeftSurfType": LeftSurfType,
                    "InitLeftSurfTangentX": Sl0_TangentX, "InitLeftSurfTangentY": Sl0_TangentY, "InitLeftSurfNorm": Sl0_Norm, "InitLeftSurfOrientation": Sl0_Orientation,
                    "InitRightSurfVertice": Sr0, "InitRightSurfType": RightSurfType,
                    "InitRightSurfTangentX": Sr0_TangentX, "InitRightSurfTangentY": Sr0_TangentY, "InitRightSurfNorm": Sr0_Norm, "InitRightSurfOrientation": Sr0_Orientation,
                    "ContactSurfsVertice": ContactSurfsVertice,
                    "ContactSurfsHalfSpace": ContactSurfsHalfSpace,
                    "ContactSurfsTypes": ContactSurfsTypes,
                    "ContactSurfsNames": ContactSurfsNames,
                    "ContactSurfsTangentX": ContactSurfsTangentX,
                    "ContactSurfsTangentY": ContactSurfsTangentY,
                    "ContactSurfsNorm": ContactSurfsNorm,
                    "ContactSurfsOrientation": ContactSurfsOrientation,
                    "AllPatchesVertices": AllPatches}

    return TerrainModel

# Elevation change with respect to an ajacent patch


def elevation_shift_wrt_adjacent_patch(current_patch=None, previous_patch=None, elevation_shift=None):
    # Get the Center of the previous patch first
    center_x, center_y, center_z = getCenter(previous_patch)

    # make lifted patch
    elevatedPatch = current_patch + np.array([[0.0, 0.0, elevation_shift],
                                              [0.0, 0.0, elevation_shift],
                                              [0.0, 0.0, elevation_shift],
                                              [0.0, 0.0, elevation_shift]])

    return elevatedPatch

# Modify Terrain (for only a single patch)
# ModificationMode = 1) rotate


def terrain_modify(TerrainModel=None, PatchIdx_for_Mod=None, ModificationMode=None, RotateAxis=None, RotateAngle=None, min_theta=None, max_theta=None):

    # store useful info
    useful_info = {}

    if TerrainModel == None:
        raise Exception("Terrain Model for Modification now provided")

    # Make a deep copy of the terrain model
    TerrainModel_Modified = copy.deepcopy(TerrainModel)

    if PatchIdx_for_Mod == None:
        raise Exception("Patch Index for Modification is not Provided")

    if ModificationMode == None:
        raise Exception("Un-Specified Modification Mode")
    elif ModificationMode == "rotate":
        # Rotate the patch
        print("We rotate the patch ", PatchIdx_for_Mod)

        # Define Rotation Axis
        if RotateAxis == None:
            rotation_type = np.random.choice(
                ["X_positive", "X_negative", "Y_positive", "Y_negative"], 1)
        else:
            rotation_type = RotateAxis
        print("Rotation Type: ", rotation_type)

        # Define Rotation Angle
        if RotateAngle == None:
            if min_theta == None or max_theta == None:
                raise Exception("Un provided Min and Max Theta")
            rotation_angle = np.round(
                np.random.uniform(min_theta, max_theta), 3)

        # Locate Rotation Patch
        patch_for_rotation = TerrainModel["ContactSurfsVertice"][PatchIdx_for_Mod]

        # make 2d Projection of the Patch
        patch_proj = copy.deepcopy(patch_for_rotation)
        patch_proj[0][-1] = 0.0
        patch_proj[1][-1] = 0.0
        patch_proj[2][-1] = 0.0
        patch_proj[3][-1] = 0.0

        # Rotate with Rotate Function and get other quantities
        Patch_rotated = rotate_patch(
            surf=patch_proj, PatchType=rotation_type, theta=rotation_angle)
        Patch_rotated_halfspace = np.concatenate(
            convert_surface_to_inequality(Patch_rotated.T), axis=None)
        Patch_rotated_TangentX, Patch_rotated_TangentY, Patch_rotated_Norm, Patch_rotated_Orientation = getTerrainTagentsNormOrientation(
            Patch_rotated)

        # Update Modified Terrain
        TerrainModel_Modified["ContactSurfsVertice"][PatchIdx_for_Mod] = Patch_rotated
        TerrainModel_Modified["ContactSurfsHalfSpace"][PatchIdx_for_Mod] = Patch_rotated_halfspace
        TerrainModel_Modified["ContactSurfsTypes"][PatchIdx_for_Mod] = rotation_type
        TerrainModel_Modified["ContactSurfsTangentX"][PatchIdx_for_Mod] = Patch_rotated_TangentX
        TerrainModel_Modified["ContactSurfsTangentY"][PatchIdx_for_Mod] = Patch_rotated_TangentY
        TerrainModel_Modified["ContactSurfsNorm"][PatchIdx_for_Mod] = Patch_rotated_Norm
        TerrainModel_Modified["ContactSurfsOrientation"][PatchIdx_for_Mod] = Patch_rotated_Orientation

        TerrainModel_Modified["AllPatchesVertices"][PatchIdx_for_Mod +
                                                    2] = Patch_rotated
        #TerrainModel_Modified["AllPatchesVertices"] = [TerrainModel_Modified["InitLeftSurfVertice"], TerrainModel_Modified["InitRightSurfVertice"], TerrainModel_Modified["ContactSurfsVertice"]]

        useful_info["rotation_angle"] = rotation_angle
        useful_info["rotation_type"] = rotation_type

    return TerrainModel_Modified, useful_info
# ----------------------------------------
# LEGACY FUNCTIONS


def rectan_gen(PatchColumn="left", PatchType="flat", ref_x=0, ref_y=0, ref_z=0, Proj_Length=0.6, Proj_Width=0.6,  theta=None, min_theta=0.08, max_theta=0.3, CenterShift=np.array([0.0, 0.0, 0.0])):
    #PatchColum = "left" or "right"
    #PatchType = "flat" or "X_positive" or "X_negative" or "Y_positive" or "Y_negative"

    # randomly Sample a rotation angle (in case of rotation patch), ip theta is not provided
    if theta == None:
        theta = np.random.uniform(min_theta, max_theta)

    # Calculate Projected Vertices
    # For Left Patch
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4
    # ref_x,ref_y,ref_z (ref_z already add shifts)
    if PatchColumn == "left":
        Proj_p1 = [ref_x + Proj_Length, ref_y + Proj_Width, ref_z]
        Proj_p2 = [ref_x,               ref_y + Proj_Width, ref_z]
        Proj_p3 = [ref_x,               ref_y,              ref_z]
        Proj_p4 = [ref_x + Proj_Length, ref_y,              ref_z]

    # For Right Patch
    #ref_x, ref_y
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4
    elif PatchColumn == "right":
        Proj_p1 = [ref_x + Proj_Length, ref_y,                           ref_z]
        Proj_p2 = [ref_x,               ref_y,                           ref_z]
        Proj_p3 = [ref_x,               ref_y - Proj_Width,              ref_z]
        Proj_p4 = [ref_x + Proj_Length, ref_y - Proj_Width,              ref_z]

    # Get Update z position of each patch

    # Copy projected patch to output patch
    p1 = Proj_p1
    p2 = Proj_p2
    p3 = Proj_p3
    p4 = Proj_p4

    # Shift the center if we have
    p1 = p1 + CenterShift
    p2 = p2 + CenterShift
    p3 = p3 + CenterShift
    p4 = p4 + CenterShift

    if PatchType == "flat":
        print("TerrainGen: flat patch - no changes to projected patch")
    elif PatchType == "X_positive":
        # theta = rotation angle, defined in radius
        #theta = np.arctan2(0.3,1)
        print("TerrainGen Rotation along X positive, with ",
              str(theta/np.pi*180), "degrees")
        delta_z = Proj_Width/2*np.tan(theta)
        # p2---------------------p1
        # |                      |
        # |                      |   ------>x
        # |                      |
        # p3---------------------p4
        p1[2] = p1[2] + delta_z
        p2[2] = p2[2] + delta_z
        p3[2] = p3[2] - delta_z
        p4[2] = p4[2] - delta_z

    elif PatchType == "X_negative":
        # theta = rotation angle, defined in radius
        #theta = np.arctan2(0.3,1)
        print("TerrainGen Rotation along X negative, with ",
              str(theta/np.pi*180), "degrees")
        delta_z = Proj_Width/2*np.tan(theta)
        # p2---------------------p1
        # |                      |
        # |                      |   ------>x
        # |                      |
        # p3---------------------p4
        p1[2] = p1[2] - delta_z
        p2[2] = p2[2] - delta_z
        p3[2] = p3[2] + delta_z
        p4[2] = p4[2] + delta_z

    elif PatchType == "Y_positive":
        # theta = rotation angle, defined in radius
        #theta = np.arctan2(0.3,1)
        print("TerrainGen Rotation along Y positive, with ",
              str(theta/np.pi*180), "degrees")
        delta_z = Proj_Length/2*np.tan(theta)
        # p2---------------------p1
        # |                      |
        # |                      |
        # |                      |
        # p3---------------------p4
        #           ^y
        p1[2] = p1[2] - delta_z
        p2[2] = p2[2] + delta_z
        p3[2] = p3[2] + delta_z
        p4[2] = p4[2] - delta_z
    elif PatchType == "Y_negative":
        #theta = np.arctan2(0.3,1)
        print("TerrainGen Rotation along Y negative, with ",
              str(theta/np.pi*180), "degrees")
        delta_z = Proj_Length/2*np.tan(theta)
        # p2---------------------p1
        # |                      |
        # |                      |
        # |                      |
        # p3---------------------p4
        #           ^y
        p1[2] = p1[2] + delta_z
        p2[2] = p2[2] - delta_z
        p3[2] = p3[2] - delta_z
        p4[2] = p4[2] + delta_z
    else:
        raise Exception("Unknown Patch Type")

    Surface = np.array([p1, p2, p3, p4])

    return Surface

# Rotate patches
#PatchColum = "left" or "right"
#PatchType = "flat" or "X_positive" or "X_negative" or "Y_positive" or "Y_negative"


def rotate_patch_ABANDONED(surf=None, PatchColumn="left", PatchType=None, theta=None, PatchShift_wrt_RotationAxis=0.0, min_theta=0.08, max_theta=0.3):
    # get the flat patch
    rotatedPatch = surf

    # randomly Sample a rotation angle (in case of rotation patch), ip theta is not provided
    if theta == None:
        theta = np.random.uniform(min_theta, max_theta)

    # Calculate (current) Patch Length and Width
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4
    # Patch Length = abs([p1][x] - [p2][x])
    patchLength = np.abs(surf[0][0] - surf[1][0])
    # Patch Width = abs([p1][y] - [p4][y])
    patchWidth = np.abs(surf[0][1] - surf[3][1])

    # Rotate Patches
    if PatchType == "flat":
        print("TerrainGen: flat patch - no changes to projected patch")

    elif "X" in PatchType:  # Rotation around X axis
        # rotate around x axis, we need to change z coordinate of borders parallel to y
        #         (p3,p4)---RightContact_patch----(p1,p2)       centaral(x_global), pointing OUTWARD        (p3,p4)------LeftContact_patch-----(p1,p2)
        #  <----outward  (local central y = 0)  inward---->                                              <----inward  (local central y = 0)  outward---->
        #  --------------------->
        #  y-axis  direction
        #  NOTE: We VIEW towards the BEGINNING of the terrain, then we have y-z in this set up:
        #  ^ (z)
        #  |
        #  |
        #  --------> (y)

        # NOTE: the horizontal lines represent the WIDTH of the patch

        # Shift Border Coordinates
        if PatchColumn == "left":
            # If LEFT CONTACT Patch (NOTE:RIGHT hand side of the figure above),
            # then the outer border coordinate (width) is POSITIVE, inner border coordinate (width) is NEGATIVE
            WidthOuterBorder = patchWidth/2.0
            WidthInnerBorder = -patchWidth/2.0
            # If Left patch move outward (POSITIVE shift) with respect to rotation axis, then the outer and inner border coordinates (width) ADD the shift
            # If Left patch move inward  (NEGATIVE shift) with respect to rotation axis,  then the outer and inner border coordinates (width) MINUS the shift
            # To UNIFY, we ADD the shift to the Border coordinates
            WidthOuterBorder = WidthOuterBorder + PatchShift_wrt_RotationAxis
            WidthInnerBorder = WidthInnerBorder + PatchShift_wrt_RotationAxis
        elif PatchColumn == "right":
            # If RIGHT CONTACT Patch (NOTE:Left hand side of the figure above),
            # then the outer border coordinate (width) is NEGATIVE, inner border coordinate (width) is POSITIVE
            WidthOuterBorder = -patchWidth/2.0
            WidthInnerBorder = patchWidth/2.0
            # If Right Patch move outward (POSITVE shift) with respect to rotation axis, then the outer and inner border coordinates (width) MINUS the shift
            # If Right patch move inward (NEGATIVE shift) with respect to rotation axis, then the outer and inner border coordinates (wdith) ADD the shift
            # To UNIFY, we MINUS the shift to the Border coordinates
            WidthOuterBorder = WidthOuterBorder - PatchShift_wrt_RotationAxis
            WidthInnerBorder = WidthInnerBorder - PatchShift_wrt_RotationAxis
        else:
            raise Exception("Unknow Patch Column Indicator")

        # Rotation to Get z coordinate
        if PatchType == "X_positive":
            # p2---------------------p1
            # |                      |
            # |                      |
            # |                      |
            # p3---------------------p4

            #      ^ (z)
            #      |  -(outer)
            #      | -
            #      |-   (theta)
            #      --------> (y)    NOTE: x is pointing outwards, we are looking towards the beginning of the terrain
            #     -
            #    -
            #   (inner)

            deltaZ_InnerBorder = np.tan(np.pi+theta)*WidthInnerBorder
            deltaZ_OuterBorder = np.tan(theta)*WidthOuterBorder

            print("delta Z xpositive", deltaZ_OuterBorder)

            # deltaZ already has positive or negative signs
            if PatchColumn == "left":
                #      ^ (z)
                #      |  -(p1,p2)(outer)
                #      | -
                #      |-   (theta)
                #      --------> (y)    NOTE: x is pointing outwards, we are looking towards the beginning of the terrain
                #     -
                #    -
                #   (p3,p4)(inner)

                rotatedPatch[0][2] = rotatedPatch[0][2] + \
                    deltaZ_OuterBorder  # p1
                rotatedPatch[1][2] = rotatedPatch[1][2] + \
                    deltaZ_OuterBorder  # p2
                rotatedPatch[2][2] = rotatedPatch[2][2] + \
                    deltaZ_InnerBorder  # p3
                rotatedPatch[3][2] = rotatedPatch[3][2] + \
                    deltaZ_InnerBorder  # p4

            elif PatchColumn == "right":
                #      ^ (z)
                #      |  -(p1,p2)(inner)
                #      | -
                #      |-   (theta)
                #      --------> (y)    NOTE: x is pointing outwards, we are looking towards the beginning of the terrain
                #     -
                #    -
                #   (p3,p4)(outer)

                rotatedPatch[0][2] = rotatedPatch[0][2] + \
                    deltaZ_InnerBorder  # p1
                rotatedPatch[1][2] = rotatedPatch[1][2] + \
                    deltaZ_InnerBorder  # p2
                rotatedPatch[2][2] = rotatedPatch[2][2] + \
                    deltaZ_OuterBorder  # p3
                rotatedPatch[3][2] = rotatedPatch[3][2] + \
                    deltaZ_OuterBorder  # p4

        elif PatchType == "X_negative":
            # p2---------------------p1
            # |                      |
            # |                      |
            # |                      |
            # p3---------------------p4

            #            ^ (z)
            # (inner) -   |
            #         -  |
            # theta    - |
            # --------------------> (y)    NOTE: x is pointing outwards, we are looking towards the beginning of the terrain
            #            |-
            #            | -
            #            |  -(outer)

            deltaZ_InnerBorder = np.tan(np.pi-theta)*WidthInnerBorder
            deltaZ_OuterBorder = np.tan(-theta)*WidthOuterBorder

            print("delta Z, x negative", deltaZ_OuterBorder)

            # deltaZ already has positive or negative signs
            if PatchColumn == "left":
                #            ^ (z)
                # (inner) -   |
                # (p3,p4)  -  |
                # theta    - |
                # --------------------> (y)    NOTE: x is pointing outwards, we are looking towards the beginning of the terrain
                #            |-
                #            | -
                #            |  -(outer)(p1,p2)
                rotatedPatch[0][2] = rotatedPatch[0][2] + \
                    deltaZ_OuterBorder  # p1
                rotatedPatch[1][2] = rotatedPatch[1][2] + \
                    deltaZ_OuterBorder  # p2
                rotatedPatch[2][2] = rotatedPatch[2][2] + \
                    deltaZ_InnerBorder  # p3
                rotatedPatch[3][2] = rotatedPatch[3][2] + \
                    deltaZ_InnerBorder  # p4

            elif PatchColumn == "right":
                #            ^ (z)
                # (outer) -   |
                # (p3,p4)  -  |
                # theta    - |
                # --------------------> (y)    NOTE: x is pointing outwards, we are looking towards the beginning of the terrain
                #            |-
                #            | -
                #            |  -(inner)(p1,p2)
                rotatedPatch[0][2] = rotatedPatch[0][2] + \
                    deltaZ_InnerBorder  # p1
                rotatedPatch[1][2] = rotatedPatch[1][2] + \
                    deltaZ_InnerBorder  # p2
                rotatedPatch[2][2] = rotatedPatch[2][2] + \
                    deltaZ_OuterBorder  # p3
                rotatedPatch[3][2] = rotatedPatch[3][2] + \
                    deltaZ_OuterBorder  # p4

    elif "Y" in PatchType:  # Rotation around Y axis
        # View along the positive y-axis

        # p2---------------------p1
        # |                      |
        # |                      |
        # |                      |
        # p3---------------------p4

        # Then we get the view, NOTE: The same for both LEFT and RIGHT CONTACT patch
        #            ^ (z)
        #            |
        #            |
        #            |
        # --------------------> (x)    NOTE: x is pointing outwards, we are looking towards the beginning of the terrain, y pointing inwards
        # (Inward)   |     (Outward)
        # (Inner)    |     (Outer)
        # (p2,p3)    |     (p1,p4)

        # Compute Length Border Coordinate (along x-axis)
        # Same for both LEFT and RIGHT CONTACT Patch, Moving Outwards (towards the end of the terrain, positive x),
        # or Moving Inward (toward the begining of the terrain, negative x)
        # NOTE: BOTH ADD the shift
        # Inner Border                                                               Outer Border
        LengthInnerBorder = -patchLength/2.0 + PatchShift_wrt_RotationAxis
        LengthOuterBorder = patchLength/2.0 + PatchShift_wrt_RotationAxis

        # Decide delta Z based on rotation angle and direction
        if PatchType == "Y_positive":

            # (p2,p3) -   ^ (z)
            #         -  |
            #          - |
            #   theta   -|
            # --------------------> (x)    NOTE: x is pointing outwards, we are looking towards the beginning of the terrain
            # (Inward)   |-     (Outward)
            # (Inner)    | -    (Outer)
            #            |  -(p1,p4)

            deltaZ_InnerBorder = np.tan(np.pi-theta)*LengthInnerBorder
            deltaZ_OuterBorder = np.tan(-theta)*LengthOuterBorder

        elif PatchType == "Y_negative":
            #            ^ (z)
            #            |  - (p1,p4)
            #            | -
            #            |- theta
            # --------------------> (x)    NOTE: x is pointing outwards, we are looking towards the beginning of the terrain
            # (Inward)  -|      (Outward)
            # (Inner)  - |      (Outer)
            # (p2,p3) -  |

            deltaZ_InnerBorder = np.tan(np.pi + theta)*LengthInnerBorder
            deltaZ_OuterBorder = np.tan(theta)*LengthOuterBorder

        # Shift z coordinate
        rotatedPatch[0][2] = rotatedPatch[0][2] + \
            deltaZ_OuterBorder  # p1 (outer)
        rotatedPatch[1][2] = rotatedPatch[1][2] + \
            deltaZ_InnerBorder  # p2 (inner)
        rotatedPatch[2][2] = rotatedPatch[2][2] + \
            deltaZ_InnerBorder  # p3 (inner)
        rotatedPatch[3][2] = rotatedPatch[3][2] + \
            deltaZ_OuterBorder  # p4 (outer)

    return rotatedPatch
