# At present, we assume the contact sequence starts from the left foot contact

import numpy as np
from sl1m.constants_and_tools import *
import matplotlib.pyplot as plt  # Matplotlib
from mpl_toolkits.mplot3d import Axes3D
from multicontact_learning_local_objectives.python.terrain_create.geometry_utils import *

# NOTE: ALL FUNCTIONS ONLY CONSIDER LEFT SWING IN THE FIRST STEP

#   Rotation angle input is radius
#       fixed_inclination: the terrain inclinations of "normal patches"
#       fixed_inclination = None: Random Patches
#       fixed_inclination = any: terrain incliation for all surfaces for antfamr/darpa terrain; terrain inclination
#       large_slope_flag = True/False
#       Num_large_slopes = Specified / None; When None, Num_large_slopes is TotalNumSurfs//10 - 1
#       large_slopes_index: a list of number to identify the which slope is the large slope
#       large_slope_directions: specify the large slope orientation (A LIST contrains X_negative, X_positive, Y_negative, Y_positive) or NONE (Randomly select)
#       large_slope_inclinations: specify the large slope inclinaton (A LIST containts radius) or NONE (random sampling)
#       large_slope_center_shifts: can specify a list of [x,y,z], or None (random generate) NOTE(NOT implemented) or Default [0,0,0]
#       large_slope_rotation_axis_shifts: can specify a list of numbers (shift rotation axis along the perpendicular axis of the rotation axis,
#       i.e. if rotate around x, then shift the rotation axis along y)


def terrain_model_gen(terrain_name=None, fixed_inclination=None,
                      random_surfsize=False, max_Proj_Length=0.6, min_Proj_Length=0.5, max_Proj_Width=0.6, min_Proj_Width=0.3,
                      Proj_Length=0.6, Proj_Width=0.6,
                      NumSteps=None, NumLookAhead=None,
                      large_slope_flag=False, Num_large_slopes=1, large_slopes_index=[8],
                      large_slope_directions=["X_negative"], large_slope_inclinations=[18.0/180.0*np.pi],
                      large_slope_center_shifts=[[0.0, 0.0, 0.0]]*1000, large_slope_rotation_axis_shifts=[[0.0, 0.0]]*1000):
    # Surf Vertex Identification
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4

    # ---------------
    # Number of Surfaces based on Number of steps and Number of lookahead
    # For contact surfaces only (not included intial two contact patches)
    TotalNumSurfs = NumSteps + NumLookAhead - 1

    # ---------------
    # Some Initial checks on the large slope
    if np.max(large_slopes_index) > TotalNumSurfs:
        raise Exception(
            "large slope index larger than the total number of surfaces")
    if (len(large_slopes_index) != len(large_slope_directions)) and (len(large_slope_directions) != len(large_slope_inclinations)) and (len(large_slope_inclinations) != Num_large_slopes):
        raise Exception("Inconsistent Large Slope setups")

    # ---------------
    # Make Initial Contact Patches (Current Define as flat patches)
    InitContactSurf_x_max = 0.2

    print("Left Init Contact Surface:")
    Sl0 = np.array([[InitContactSurf_x_max, 0.6, 0.], [-0.2, 0.6, 0.],
                   [-0.2, 0.0, 0.], [InitContactSurf_x_max, 0.0, 0.0]])
    LeftSurfType = getSurfaceType(Sl0)  # "flat"
    print("Left Init Contact Surface Type: ", LeftSurfType)
    Sl0_TangentX, Sl0_TangentY, Sl0_Norm, Sl0_Orientation = getTerrainTagentsNormOrientation(
        Sl0)

    print("Right Init Contact Surface")
    Sr0 = np.array([[InitContactSurf_x_max, 0, 0.], [-0.2, 0, 0.],
                   [-0.2, -0.6, 0.], [InitContactSurf_x_max, -0.6, 0.0]])
    RightSurfType = getSurfaceType(Sr0)  # "flat"
    print("Right Init Contact Surface Type: ", RightSurfType)
    Sr0_TangentX, Sr0_TangentY, Sr0_Norm, Sr0_Orientation = getTerrainTagentsNormOrientation(
        Sr0)

    # ------------
    # Make Contact Patches
    #   Decide Terrain Pattern
    if (terrain_name == "flat") or (terrain_name == "stair") or (terrain_name == "single_flat") or (terrain_name == "flat_with_large_slope"):
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
        TerrainTypeList = ["flat", "X_positive", "X_negative",
                           "Y_positive", "Y_negative"]  # 5 types of terrain
        np.random.seed()
        terrain_pattern = np.random.choice(TerrainTypeList, 200)
    else:
        raise Exception("Unknown Terrain Type")

    #   Create Containers for Describing Contacts
    ContactSurfsVertice = []
    ContactSurfsHalfSpace = []
    ContactSurfsTypes = []
    ContactSurfsNames = []
    ContactSurfsTangentX = []
    ContactSurfsTangentY = []
    ContactSurfsNorm = []
    ContactSurfsOrientation = []
    AllPatches = []

    # Below is for generating sparse patches
    for surfNum in range(TotalNumSurfs):

        print("Put Contact Patch ", surfNum)

        # Decide Surf size
        if random_surfsize == True:  # update surf size, if random surf size is selected
            Proj_Length = np.random.uniform(min_Proj_Length, max_Proj_Length)
            Proj_Width = np.random.uniform(min_Proj_Length, max_Proj_Length)

        if surfNum == 0:
            # (Currently) Make a Left Stepping Surf at the first step (surfNum = 0)
            # Need to construct ref_point coordinate (P3)
            # ---- #p2---------------------p1
            # |                      |
            # Sl0  # |         S0           |
            # |                      |
            # ---- #p3 (ref_x, ref_y)-------p4
            # |                      |
            # Sr0  # |                      |
            # |                      |

            # Define which Contact foot is for the Surface
            SurfContactLeft = "left"

            # Special case need to consider Initial Left and Right Patches
            ref_x = Sl0[0][0]  # ref_x decided by the P1_x of Sl0
            ref_y = Sr0[0][1]  # ref_y decided by the P1_y of Sr0
            center = getCenter(Sr0)
            # ref_z decided by the central z of Sr0 (relative foot height constraints is enforced with respect to stationary foot)
            ref_z = center[2]

        elif surfNum == 1:  # Special Case needs to consider Initial Right and the first contact patch
            # (Currently) The first step (surfNum = 0) make a left step, the second step (surfNum = 1) makes a right step
            # Need to construct ref_point coordinate (P2)
            # ---- #  ---------------------
            # |                      |
            # |         S0           |
            # |                      |
            # ---- #  ---------------------
            # P2 (ref_x, ref_y)------P1
            # |                      |
            # Sr0  # |         S1           |
            # |                      |
            # P3---------------------P4

            # Define which Contact foot is for the Surface
            SurfContactLeft = "right"

            # Special case need to consider Initial Right and The First Contact Surf(S0)
            ref_x = Sr0[0][0]  # ref_x decided by the P1_x of Sr0
            # ref_y decided by the P3_y of S0 ( ContactSurfsVertice[0])
            ref_y = ContactSurfsVertice[0][2][1]
            center = getCenter(ContactSurfsVertice[0])
            # ref_z decided by the central z of S0 (relative foot height constraints is enforced with respect to stationary foot)
            ref_z = center[2]

        else:
            if surfNum % 2 == 0:  # Even number of steps (Swing the Left)
                # ---- #p2---------------------p1
                # |                      |
                # Sn-2 # |         Sn           |
                # (L) # |        (L)           |
                # ---- #p3 (ref_x, ref_y)-------p4
                # |                      |
                # Sn-1 # |                      |
                # (R) # |                      |

                # Define which Contact foot is for the Surface
                SurfContactLeft = "left"

                # ref_x decided by the P1_x of Sn-2 (-2 in index)
                ref_x = ContactSurfsVertice[-2][0][0]
                # ref_y decided by the P1_y of Sn-1 (-1 in index)
                ref_y = ContactSurfsVertice[-1][0][1]
                center = getCenter(ContactSurfsVertice[-1])
                # ref_z decided by the central z of Sn-1 (relative foot height constraints is enforced with respect to stationary foot)
                ref_z = center[2]

            elif surfNum % 2 == 1:  # Odd number of steps
                # ---- #  ---------------------
                # |                      |
                # |         Sn-1         |
                # |          (L)         |
                # ---- #  ---------------------
                # P2 (ref_x, ref_y)------P1
                # |                      |
                # Sn-2 # |         Sn           |
                # (R) # |        (R)           |
                # P3---------------------P4

                # Define which Contact foot is for the Surface
                SurfContactLeft = "right"

                # ref_x decided by the P1_x of Sn-2
                ref_x = ContactSurfsVertice[-2][0][0]
                # ref_y decided by the P3_y of Sn-1 (ContactSurfsVertice[-1])
                ref_y = ContactSurfsVertice[-1][2][1]
                center = getCenter(ContactSurfsVertice[-1])
                # ref_z decided by the central z of Sn-1 (relative foot height constraints is enforced with respect to stationary foot)
                ref_z = center[2]

        # Creating Patches
        # Special treatment for the large slope
        if (large_slope_flag == True) and (surfNum in large_slopes_index):
            seqnum_in_large_slope = large_slopes_index.index(surfNum)
            surf_temp = rectan_gen(PatchColumn=SurfContactLeft, PatchType=large_slope_directions[seqnum_in_large_slope],
                                   ref_x=ref_x, ref_y=ref_y, ref_z=ref_z, Proj_Length=Proj_Length, Proj_Width=Proj_Width,
                                   theta=large_slope_inclinations[seqnum_in_large_slope], CenterShift=np.array([0., 0., 0.]))
        # For normal patches
        else:
            # NOTE: needs to be randomized in the future
            centershift = np.array([0.0, 0.0, 0.0])
            surf_temp = rectan_gen(PatchColumn=SurfContactLeft, PatchType=terrain_pattern[surfNum],
                                   ref_x=ref_x, ref_y=ref_y, ref_z=ref_z, Proj_Length=Proj_Length, Proj_Width=Proj_Width,  theta=fixed_inclination)

        HalfSpace_temp = np.concatenate(
            convert_surface_to_inequality(surf_temp.T), axis=None)
        TangentX_temp, TangentY_temp, Norm_temp, OrientationTemp = getTerrainTagentsNormOrientation(
            surf_temp)

        ContactSurfsVertice.append(surf_temp)
        ContactSurfsHalfSpace.append(HalfSpace_temp)
        ContactSurfsNames.append("S"+str(surfNum))
        ContactSurfsTypes.append(terrain_pattern[surfNum])
        ContactSurfsTangentX.append(TangentX_temp)
        ContactSurfsTangentY.append(TangentY_temp)
        ContactSurfsNorm.append(Norm_temp)
        ContactSurfsOrientation.append(OrientationTemp)

    # Get all Patches
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
                    "ContactSurfsTangentX": ContactSurfsTangentX,
                    "ContactSurfsTangentY": ContactSurfsTangentY,
                    "ContactSurfsNorm": ContactSurfsNorm,
                    "ContactSurfsOrientation": ContactSurfsOrientation,
                    "AllPatchesVertices": AllPatches}

    return TerrainModel


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
        slope = 0.325
        SlopePatch_Vertice = np.array([[1.8, 1, slope], [1.5, 1, slope], [
                                      1.5, -1.0, -slope], [1.8, -1.0, -slope]])
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

    elif terrain_name == "flat":

        # Define Initial Patches
        print("Left Init Contact Surface:")
        # initial patch for the left foot (for the first step only)
        Sl0 = np.array([[10, 1, 0.], [-1, 1, 0.], [-1, -1, 0.], [10, -1, 0.]])
        LeftSurfType = getSurfaceType(Sl0)  # "flat"
        print("Left Init Contact Surface Type: ", LeftSurfType)
        Sl0_TangentX, Sl0_TangentY, Sl0_Norm, Sl0_Orientation = getTerrainTagentsNormOrientation(
            Sl0)

        print("Right Init Contact Surface")
        # initial patch for the right foot (for the first step only)
        Sr0 = np.array([[10, 1, 0.], [-1, 1, 0.], [-1, -1, 0.], [10, -1, 0.]])
        RightSurfType = getSurfaceType(Sr0)  # "flat"
        print("Right Init Contact Surface Type: ", RightSurfType)
        Sr0_TangentX, Sr0_TangentY, Sr0_Norm, Sr0_Orientation = getTerrainTagentsNormOrientation(
            Sr0)

        FlatPatch_Vertice = np.array(
            [[10, 1, 0.], [-1, 1, 0.], [-1, -1, 0.], [10, -1, 0.]])
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
