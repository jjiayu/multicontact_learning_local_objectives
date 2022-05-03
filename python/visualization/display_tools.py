import numpy as np
import matplotlib.pyplot as plt  # Matplotlib
from mpl_toolkits.mplot3d import Axes3D
import multicontact_learning_local_objectives.python.terrain_create.geometry_utils as geometric_utils
import pickle
from multicontact_learning_local_objectives.python.utils import *

# Draw a single surface in the 3D scene


def drawSurface(Surface=None, ax=None):
    # Make the Rectangle, start from the top right coner, and move counter clockwise
    SurfRect = np.append(Surface, [Surface[0]], axis=0)

    cx = [c[0] for c in SurfRect]
    cy = [c[1] for c in SurfRect]
    cz = [c[2] for c in SurfRect]
    ax.plot(cx, cy, cz)

    return ax


def drawFootPatch(P=None, P_TangentX=None, P_TangentY=None, line_color=None, LineWidth=2, LineType = 'solid', footlength = 0.2, footwidth = 0.1, ax=None):
    # Foot Vertex Location Assignment
    # P3----------------P1
    # |                  |
    # |                  |
    # |                  |
    # P4----------------P2

    # Contact Points
    P1 = P + footlength/2.0*P_TangentX + footwidth/2.0*P_TangentY
    P2 = P + footlength/2.0*P_TangentX - footwidth/2.0*P_TangentY
    P3 = P - footlength/2.0*P_TangentX + footwidth/2.0*P_TangentY
    P4 = P - footlength/2.0*P_TangentX - footwidth/2.0*P_TangentY

    # make the rectangle
    FootPatch = [P1, P3, P4, P2]

    FootRectan = np.append(FootPatch, [FootPatch[0]], axis=0)

    cx = [c[0] for c in FootRectan]
    cy = [c[1] for c in FootRectan]
    cz = [c[2] for c in FootRectan]
    ax.plot(cx, cy, cz, color=line_color, linewidth=LineWidth, linestyle = LineType)

    return ax

# Show the complete Terrain Model
#   The first element is the left init patch, the second element is the right init patch, all the rest are the contact patches


def drawTerrain(Sl0surf=None, Sr0surf=None, ContactSurfs=None, printTerrainVertice=True, fig=None, ax=None, EndPatchNum=30):

    # Draw Initial Left Contact Surface
    if printTerrainVertice == True:
        print("Patches:")
        print("Surface::left init patch: \n", Sl0surf)
    ax = drawSurface(Surface=Sl0surf, ax=ax)

    # Draw Initial Right Contact Surface
    if printTerrainVertice == True:
        print("Surface::right init patch: \n", Sr0surf)
    ax = drawSurface(Surface=Sr0surf, ax=ax)

    # Draw Contact Patches
    surf_cnt = 0
    for surf in ContactSurfs:
        # Draw Patch
        ax = drawSurface(Surface=surf, ax=ax)

        if printTerrainVertice == True:
            print("Surface::Step ", str(surf_cnt),
                  ": \n", surf)  # count from 0

        # update surface count
        surf_cnt = surf_cnt + 1

    # Get x limit
    #x_far = ContactSurfs[EndPatchNum][0][0]+0.5
    #ax.set_xlim3d(-1.0, x_far)

    return ax

# Put Label


def labelSurface(Sl0surf=None, Sr0surf=None, ContactSurfs=None, fig=None, ax=None):

    # Get Center of Init Left Patch
    Sl0_center_x, Sl0_center_y, Sl0_center_z = geometric_utils.getCenter(
        Surface=Sl0surf)
    # Label Init Left Patch Center
    ax.text(Sl0_center_x, Sl0_center_y, Sl0_center_z, 'SL0')

    # Get Center of Init Right Patch
    Sr0_center_x, Sr0_center_y, Sr0_center_z = geometric_utils.getCenter(
        Surface=Sr0surf)
    # Label Init Right Patch Center
    ax.text(Sr0_center_x, Sr0_center_y, Sr0_center_z, 'SR0')

    # Label All Other Patches
    surf_cnt = 0
    for surf in ContactSurfs:
        # Get patch center
        center_x, center_y, center_z = geometric_utils.getCenter(Surface=surf)
        # Label center
        ax.text(center_x, center_y, center_z, 'S'+str(surf_cnt))

        # update surface count
        surf_cnt = surf_cnt + 1

    return ax

# Draw Trajectory of a single Optimization Routine (with the entire horizon)


def drawSingleOptTraj(optResult=None, fig=None, ax=None, FootMarkerSize=4):
    # Process the First Level
    # Get First Level Result
    var_idx_lv1 = optResult["var_idx"]["Level1_Var_Index"]

    # Get Full optimization result
    x_opt = optResult["opt_res"]

    # Get CoM res x, y, z
    x_lv1_res = np.array(x_opt[var_idx_lv1["x"][0]:var_idx_lv1["x"][1]+1])
    y_lv1_res = np.array(x_opt[var_idx_lv1["y"][0]:var_idx_lv1["y"][1]+1])
    z_lv1_res = np.array(x_opt[var_idx_lv1["z"][0]:var_idx_lv1["z"][1]+1])
    # Get Contact Location
    px_lv1_res = np.array(x_opt[var_idx_lv1["px"][0]:var_idx_lv1["px"][1]+1])
    py_lv1_res = np.array(x_opt[var_idx_lv1["py"][0]:var_idx_lv1["py"][1]+1])
    pz_lv1_res = np.array(x_opt[var_idx_lv1["pz"][0]:var_idx_lv1["pz"][1]+1])

    # Plot CoM Traj
    ax.plot3D(x_lv1_res, y_lv1_res, z_lv1_res, color='blue',
              linestyle='dashed', linewidth=2, markersize=12)
    #Projection of the CoM trajectory on the Ground
    ax.plot3D(x_lv1_res, y_lv1_res, np.zeros(z_lv1_res.shape), color='blue',
              linestyle='dashed', linewidth=2, markersize=12)
    # Plot Initial Contact Location
    ax.scatter(optResult["PLx_init"], optResult["PLy_init"],
               optResult["PLz_init"], c='r', marker='o', linewidth=FootMarkerSize)
    ax.scatter(optResult["PRx_init"], optResult["PRy_init"],
               optResult["PRz_init"], c='b', marker='o', linewidth=FootMarkerSize)
    # Draw Initial Contact Patches
    # The actual footsize (larger size)
    ax = drawFootPatch(P=np.array([optResult["PLx_init"], optResult["PLy_init"], optResult["PLz_init"]]),
                       P_TangentX=optResult["PL_init_TangentX"], P_TangentY=optResult["PL_init_TangentY"], line_color='r', LineType = 'solid', footlength = 0.22, footwidth = 0.12, ax=ax)
    ax = drawFootPatch(P=np.array([optResult["PRx_init"], optResult["PRy_init"], optResult["PRz_init"]]),
                       P_TangentX=optResult["PR_init_TangentX"], P_TangentY=optResult["PR_init_TangentY"], line_color='b', LineType = 'solid', footlength = 0.22, footwidth = 0.12, ax=ax)
    # The shrinked footsize for defining contact points (smaller size)
    ax = drawFootPatch(P=np.array([optResult["PLx_init"], optResult["PLy_init"], optResult["PLz_init"]]),
                       P_TangentX=optResult["PL_init_TangentX"], P_TangentY=optResult["PL_init_TangentY"], line_color='r', LineType = 'dashed', footlength = 0.2, footwidth = 0.1, ax=ax)
    ax = drawFootPatch(P=np.array([optResult["PRx_init"], optResult["PRy_init"], optResult["PRz_init"]]),
                       P_TangentX=optResult["PR_init_TangentX"], P_TangentY=optResult["PR_init_TangentY"], line_color='b', LineType = 'dashed', footlength = 0.2, footwidth = 0.1, ax=ax)

    # Plot the Swing Foot
    if optResult["LeftSwingFlag"] == 1:
        StepColor = 'r'
    elif optResult["RightSwingFlag"] == 1:
        StepColor = 'b'
    # plot location
    ax.scatter(px_lv1_res, py_lv1_res, pz_lv1_res, c=StepColor,
               marker='o', linewidth=FootMarkerSize)
    # draw Patch for the first step (index 0 in contact lists)
    # The actual footsize (larger size)
    ax = drawFootPatch(P=np.concatenate((px_lv1_res, py_lv1_res, pz_lv1_res), axis=None),
                       P_TangentX=optResult["SurfTangentsX"][0], P_TangentY=optResult["SurfTangentsY"][0], line_color=StepColor, LineType = 'solid', footlength = 0.22, footwidth = 0.12, ax=ax)
    # The shrinked footsize for defining contact points (smaller size)
    ax = drawFootPatch(P=np.concatenate((px_lv1_res, py_lv1_res, pz_lv1_res), axis=None),
                       P_TangentX=optResult["SurfTangentsX"][0], P_TangentY=optResult["SurfTangentsY"][0], line_color=StepColor, LineType = 'dashed', footlength = 0.2, footwidth = 0.1, ax=ax)

    # Process the Second Level
    # Get opt result for the second level
    x_opt_lv2 = x_opt[var_idx_lv1["Ts"][1]+1:]
    # Get var index for the second level
    var_idx_lv2 = optResult["var_idx"]["Level2_Var_Index"]

    if var_idx_lv2:  # If we have the second level, get the optimization result
        # CoM x, y, z
        x_lv2_res = np.array(
            x_opt_lv2[var_idx_lv2["x"][0]:var_idx_lv2["x"][1]+1])
        y_lv2_res = np.array(
            x_opt_lv2[var_idx_lv2["y"][0]:var_idx_lv2["y"][1]+1])
        z_lv2_res = np.array(
            x_opt_lv2[var_idx_lv2["z"][0]:var_idx_lv2["z"][1]+1])
        # Contact locations
        px_lv2_res = np.array(
            x_opt_lv2[var_idx_lv2["px"][0]:var_idx_lv2["px"][1]+1])
        py_lv2_res = np.array(
            x_opt_lv2[var_idx_lv2["py"][0]:var_idx_lv2["py"][1]+1])
        pz_lv2_res = np.array(
            x_opt_lv2[var_idx_lv2["pz"][0]:var_idx_lv2["pz"][1]+1])

        # Plot CoM Traj
        ax.plot3D(x_lv2_res, y_lv2_res, z_lv2_res, color='green',
                  linestyle='dashed', linewidth=2, markersize=12)
        # Plot projected CoM Traj
        ax.plot3D(x_lv2_res, y_lv2_res, np.zeros(z_lv2_res.shape), color='green',
                  linestyle='dashed', linewidth=2, markersize=12)
                  #plot where the com from swing phase (num of knots = (len(x_lv1_res)-1)/3)

        # Draw the projected starting knot of the swing phase
        ax.scatter(x_lv1_res[int((len(x_lv1_res)-1)/3)], y_lv1_res[int((len(x_lv1_res)-1)/3)], 0.0, c='black',
                   marker='o', linewidth=5)

        # Draw the projected ending knot of the swing phase (starting knot of the double support phase)
        ax.scatter(x_lv1_res[int((len(x_lv1_res)-1)/3*2)], y_lv1_res[int((len(x_lv1_res)-1)/3*2)], 0.0, c='black',
                   marker='o', linewidth=5)

        # Contact Parameters of Level 2
        lv2_ContactTengentX = optResult["SurfTangentsX"][1:]
        lv2_ContactTengentY = optResult["SurfTangentsY"][1:]

        # Plot Swing Foot
        if optResult["LeftSwingFlag"] == 1:
            StepColor = 'b'
            ax.scatter(px_lv2_res[0::2], py_lv2_res[0::2], pz_lv2_res[0::2],
                       c=StepColor, marker='o', linewidth=FootMarkerSize)
            StepColor = 'r'
            ax.scatter(px_lv2_res[1::2], py_lv2_res[1::2], pz_lv2_res[1::2],
                       c=StepColor, marker='o', linewidth=FootMarkerSize)

            # Get Contact location and parameters for left and right foot
            px_lv2_right_foot = px_lv2_res[0::2]
            py_lv2_right_foot = py_lv2_res[0::2]
            pz_lv2_right_foot = pz_lv2_res[0::2]
            lv2_ContactTengentX_right_foot = lv2_ContactTengentX[0::2]
            lv2_ContactTengentY_right_foot = lv2_ContactTengentY[0::2]

            px_lv2_left_foot = px_lv2_res[1::2]
            py_lv2_left_foot = py_lv2_res[1::2]
            pz_lv2_left_foot = pz_lv2_res[1::2]
            lv2_ContactTengentX_left_foot = lv2_ContactTengentX[1::2]
            lv2_ContactTengentY_left_foot = lv2_ContactTengentY[1::2]

        elif optResult["RightSwingFlag"] == 1:
            StepColor = 'r'
            ax.scatter(px_lv2_res[0::2], py_lv2_res[0::2], pz_lv2_res[0::2],
                       c=StepColor, marker='o', linewidth=FootMarkerSize)
            StepColor = 'b'
            ax.scatter(px_lv2_res[1::2], py_lv2_res[1::2], pz_lv2_res[1::2],
                       c=StepColor, marker='o', linewidth=FootMarkerSize)

            # Get Contact location and parameters for left and right foot
            px_lv2_left_foot = px_lv2_res[0::2]
            py_lv2_left_foot = py_lv2_res[0::2]
            pz_lv2_left_foot = pz_lv2_res[0::2]
            lv2_ContactTengentX_left_foot = lv2_ContactTengentX[0::2]
            lv2_ContactTengentY_left_foot = lv2_ContactTengentY[0::2]

            px_lv2_right_foot = px_lv2_res[1::2]
            py_lv2_right_foot = py_lv2_res[1::2]
            pz_lv2_right_foot = pz_lv2_res[1::2]
            lv2_ContactTengentX_right_foot = lv2_ContactTengentX[1::2]
            lv2_ContactTengentY_right_foot = lv2_ContactTengentY[1::2]

        # Draw Foot Patch
        #   For left Foot
        for Contactnum in range(len(px_lv2_left_foot)):
            # The actual footsize (larger size)
            ax = drawFootPatch(P=np.concatenate((px_lv2_left_foot[Contactnum], py_lv2_left_foot[Contactnum], pz_lv2_left_foot[Contactnum]), axis=None),
                               P_TangentX=lv2_ContactTengentX_left_foot[Contactnum], P_TangentY=lv2_ContactTengentY_left_foot[Contactnum],
                               line_color='r', LineType = 'solid', footlength = 0.22, footwidth = 0.12,
                               ax=ax)
            # The shrinked footsize for defining contact points (smaller size)
            ax = drawFootPatch(P=np.concatenate((px_lv2_left_foot[Contactnum], py_lv2_left_foot[Contactnum], pz_lv2_left_foot[Contactnum]), axis=None),
                               P_TangentX=lv2_ContactTengentX_left_foot[Contactnum], P_TangentY=lv2_ContactTengentY_left_foot[Contactnum],
                               line_color='r', LineType = 'dashed', footlength = 0.2, footwidth = 0.1,
                               ax=ax)
        #   For Right Foot
        for Contactnum in range(len(px_lv2_right_foot)):
            # The actual footsize (larger size)
            ax = drawFootPatch(P=np.concatenate((px_lv2_right_foot[Contactnum], py_lv2_right_foot[Contactnum], pz_lv2_right_foot[Contactnum]), axis=None),
                               P_TangentX=lv2_ContactTengentX_right_foot[Contactnum], P_TangentY=lv2_ContactTengentY_right_foot[Contactnum],
                               line_color='b', LineType = 'solid', footlength = 0.22, footwidth = 0.12,
                               ax=ax)
            # The shrinked footsize for defining contact points (smaller size)
            ax = drawFootPatch(P=np.concatenate((px_lv2_right_foot[Contactnum], py_lv2_right_foot[Contactnum], pz_lv2_right_foot[Contactnum]), axis=None),
                               P_TangentX=lv2_ContactTengentX_right_foot[Contactnum], P_TangentY=lv2_ContactTengentY_right_foot[Contactnum],
                               line_color='b', LineType = 'dashed', footlength = 0.2, footwidth = 0.1,
                               ax=ax)
    # Set xlim
    if var_idx_lv2:
        ax.set_xlim3d(x_lv1_res[0]-0.2, x_lv2_res[-1]+0.5)
    else:
        ax.set_xlim3d(x_lv1_res[0]-0.2, x_lv1_res[-1]+0.5)
    return ax

# Draw Optimization Result of All Rounds/Steps
#   allOptTraj is the list of [Dict]SingleOptResult for all steps (each element is the input of the function "drawSingleOptTraj")


def DrawAllExecutionHorizon(allOptResult=None, fig=None, ax=None, FootMarkerSize=4):
    # Get First Level Result
    # Get from the first elements info
    var_idx_lv1 = allOptResult[0]["var_idx"]["Level1_Var_Index"]

    # Draw Intial Footstep Location and contact patches
    # For the initial left
    ax.scatter(allOptResult[0]["PLx_init"], allOptResult[0]["PLy_init"],
               allOptResult[0]["PLz_init"], c='r', marker='o', linewidth=FootMarkerSize)
    # The actual footsize (larger size)
    ax = drawFootPatch(P=np.array([allOptResult[0]["PLx_init"], allOptResult[0]["PLy_init"], allOptResult[0]["PLz_init"]]),
                       P_TangentX=allOptResult[0]["PL_init_TangentX"], P_TangentY=allOptResult[0]["PL_init_TangentY"], line_color='r', 
                       LineType = 'solid', footlength = 0.22, footwidth = 0.12,
                       ax=ax)
    # The shrinked footsize for defining contact points (smaller size)
    ax = drawFootPatch(P=np.array([allOptResult[0]["PLx_init"], allOptResult[0]["PLy_init"], allOptResult[0]["PLz_init"]]),
                       P_TangentX=allOptResult[0]["PL_init_TangentX"], P_TangentY=allOptResult[0]["PL_init_TangentY"], line_color='r', 
                       LineType = 'dashed', footlength = 0.2, footwidth = 0.1,
                       ax=ax)

    # For the initial right
    ax.scatter(allOptResult[0]["PRx_init"], allOptResult[0]["PRy_init"],
               allOptResult[0]["PRz_init"], c='b', marker='o', linewidth=FootMarkerSize)
    # The actual footsize (larger size)
    ax = drawFootPatch(P=np.array([allOptResult[0]["PRx_init"], allOptResult[0]["PRy_init"], allOptResult[0]["PRz_init"]]),
                       P_TangentX=allOptResult[0]["PR_init_TangentX"], P_TangentY=allOptResult[0]["PR_init_TangentY"], line_color='b', 
                       LineType = 'solid', footlength = 0.22, footwidth = 0.12,
                       ax=ax)
    # The shrinked footsize for defining contact points (smaller size)
    ax = drawFootPatch(P=np.array([allOptResult[0]["PRx_init"], allOptResult[0]["PRy_init"], allOptResult[0]["PRz_init"]]),
                    P_TangentX=allOptResult[0]["PR_init_TangentX"], P_TangentY=allOptResult[0]["PR_init_TangentY"], line_color='b', 
                    LineType = 'dashed', footlength = 0.2, footwidth = 0.1,
                    ax=ax)

    # Draw CoM and Contacts
    for roundNum in range(len(allOptResult)):
        # Decide Colors
        if roundNum % 2 == 0:  # Even number of steps
            # CoM Traj Color
            CoM_Color = 'green'

            # FootStep Color
            # Even round, and the first step swing left
            if allOptResult[0]["LeftSwingFlag"] == 1:
                P_Color = "red"
            # Even round, and the first step swing right
            elif allOptResult[0]["RightSwingFlag"] == 1:
                P_Color = "blue"

        elif roundNum % 2 == 1:  # Odd number of steps
            # CoM Traj Color
            CoM_Color = 'blue'

            # FootStep color
            # odd round, and the first step swing left
            if allOptResult[0]["LeftSwingFlag"] == 1:
                P_Color = "blue"
            # odd round, and the first step swing roight
            elif allOptResult[0]["RightSwingFlag"] == 1:
                P_Color = "red"

        # Get Full Optimization Result
        x_opt = allOptResult[roundNum]["opt_res"]
        Contact_TangentX = allOptResult[roundNum]["SurfTangentsX"][0]
        Contact_TangentY = allOptResult[roundNum]["SurfTangentsY"][0]

        # Get CoM Traj
        x_lv1_res = np.array(x_opt[var_idx_lv1["x"][0]:var_idx_lv1["x"][1]+1])
        y_lv1_res = np.array(x_opt[var_idx_lv1["y"][0]:var_idx_lv1["y"][1]+1])
        z_lv1_res = np.array(x_opt[var_idx_lv1["z"][0]:var_idx_lv1["z"][1]+1])
        # Get Contact Location
        px_lv1_res = np.array(
            x_opt[var_idx_lv1["px"][0]:var_idx_lv1["px"][1]+1])
        py_lv1_res = np.array(
            x_opt[var_idx_lv1["py"][0]:var_idx_lv1["py"][1]+1])
        pz_lv1_res = np.array(
            x_opt[var_idx_lv1["pz"][0]:var_idx_lv1["pz"][1]+1])

        # Plot CoM
        ax.plot3D(x_lv1_res, y_lv1_res, z_lv1_res, color=CoM_Color,
                  linestyle='dashed', linewidth=2, markersize=12)
        
        #plot com projection
        ax.plot3D(x_lv1_res, y_lv1_res, np.zeros(z_lv1_res.shape), color=CoM_Color,
                  linestyle='dashed', linewidth=2, markersize=12)

        #plot starting knot of the swing phase
        ax.scatter(x_lv1_res[int((len(x_lv1_res)-1)/3)], y_lv1_res[int((len(x_lv1_res)-1)/3)], 0.0, c='black',
                   marker='o', linewidth=5)

        #plot the ending knot of the swing pahse (the beginning of the post-landing phase)
        ax.scatter(x_lv1_res[int((len(x_lv1_res)-1)/3*2)], y_lv1_res[int((len(x_lv1_res)-1)/3*2)], 0.0, c='black',
                   marker='o', linewidth=5)

        # Plot Step Location
        ax.scatter(px_lv1_res, py_lv1_res, pz_lv1_res, c=P_Color,
                   marker='o', linewidth=FootMarkerSize)
        # Draw Contact Patch
        # The actual footsize (larger size)
        ax = drawFootPatch(P=np.concatenate((px_lv1_res, py_lv1_res, pz_lv1_res), axis=None),
                           P_TangentX=Contact_TangentX, P_TangentY=Contact_TangentY, line_color=P_Color, 
                           LineType = 'solid', footlength = 0.22, footwidth = 0.12,
                           ax=ax)
        # The shrinked footsize for defining contact points (smaller size)
        ax = drawFootPatch(P=np.concatenate((px_lv1_res, py_lv1_res, pz_lv1_res), axis=None),
                           P_TangentX=Contact_TangentX, P_TangentY=Contact_TangentY, line_color=P_Color, 
                           LineType = 'dashed', footlength = 0.2, footwidth = 0.1,
                           ax=ax)
    # Set xlim
    ax.set_xlim3d(allOptResult[0]["opt_res"][var_idx_lv1["x"][0]] -
                  0.2, allOptResult[-1]["opt_res"][var_idx_lv1["x"][-1]]+0.5)
    return ax

#   Draw Initial Configuration
def drawInitConfig(InitConfig=None, fig=None, ax=None, color=None, LineWidth=10):
    if color == None:
        # draw init CoM
        ax.scatter(InitConfig["x_init"], InitConfig["y_init"],
                   InitConfig["z_init"], c='g', marker='o', linewidth=LineWidth)
        
        # draw init Left contact location and patch
        ax.scatter(InitConfig["PLx_init"], InitConfig["PLy_init"],
                   InitConfig["PLz_init"], c='r', marker='o', linewidth=LineWidth)
        # The actual footsize (larger size)
        drawFootPatch(P=np.array([InitConfig["PLx_init"], InitConfig["PLy_init"], InitConfig["PLz_init"]]),
                      P_TangentX=InitConfig["PL_init_TangentX"], P_TangentY=InitConfig["PL_init_TangentY"], line_color='r', 
                      LineType = 'solid', footlength = 0.22, footwidth = 0.12,
                      ax=ax)
        # The shrinked footsize for defining contact points (smaller size)
        drawFootPatch(P=np.array([InitConfig["PLx_init"], InitConfig["PLy_init"], InitConfig["PLz_init"]]),
                      P_TangentX=InitConfig["PL_init_TangentX"], P_TangentY=InitConfig["PL_init_TangentY"], line_color='r', 
                      LineType = 'dashed', footlength = 0.2, footwidth = 0.1,
                      ax=ax)

        # draw init Right contact location
        ax.scatter(InitConfig["PRx_init"], InitConfig["PRy_init"],
                   InitConfig["PRz_init"], c='b', marker='o', linewidth=LineWidth)
        # The actual footsize (larger size)
        drawFootPatch(P=np.array([InitConfig["PRx_init"], InitConfig["PRy_init"], InitConfig["PRz_init"]]),
                      P_TangentX=InitConfig["PR_init_TangentX"], P_TangentY=InitConfig["PR_init_TangentY"], line_color='b', 
                      LineType = 'solid', footlength = 0.22, footwidth = 0.12,
                      ax=ax)
        # The shrinked footsize for defining contact points (smaller size)
        drawFootPatch(P=np.array([InitConfig["PRx_init"], InitConfig["PRy_init"], InitConfig["PRz_init"]]),
                      P_TangentX=InitConfig["PR_init_TangentX"], P_TangentY=InitConfig["PR_init_TangentY"], line_color='b', 
                      LineType = 'dashed', footlength = 0.2, footwidth = 0.1,
                      ax=ax)
    else:
        # draw init CoM
        ax.scatter(InitConfig["x_init"], InitConfig["y_init"],
                   InitConfig["z_init"], c=color, marker='o', linewidth=LineWidth)
        
        # draw init Left contact location and patch
        ax.scatter(InitConfig["PLx_init"], InitConfig["PLy_init"],
                   InitConfig["PLz_init"], c=color, marker='o', linewidth=LineWidth)
        # The actual footsize (larger size)
        drawFootPatch(P=np.array([InitConfig["PLx_init"], InitConfig["PLy_init"], InitConfig["PLz_init"]]),
                      P_TangentX=InitConfig["PL_init_TangentX"], P_TangentY=InitConfig["PL_init_TangentY"], line_color=color, 
                      LineType = 'solid', footlength = 0.22, footwidth = 0.12,
                      ax=ax)
        # The shrinked footsize for defining contact points (smaller size)
        drawFootPatch(P=np.array([InitConfig["PLx_init"], InitConfig["PLy_init"], InitConfig["PLz_init"]]),
                      P_TangentX=InitConfig["PL_init_TangentX"], P_TangentY=InitConfig["PL_init_TangentY"], line_color=color, 
                      LineType = 'dashed', footlength = 0.2, footwidth = 0.1,
                      ax=ax)

        # draw init Right contact location
        ax.scatter(InitConfig["PRx_init"], InitConfig["PRy_init"],
                   InitConfig["PRz_init"], c=color, marker='o', linewidth=LineWidth)
        # The actual footsize (larger size)
        drawFootPatch(P=np.array([InitConfig["PRx_init"], InitConfig["PRy_init"], InitConfig["PRz_init"]]),
                      P_TangentX=InitConfig["PR_init_TangentX"], P_TangentY=InitConfig["PR_init_TangentY"], line_color=color, 
                      LineType = 'solid', footlength = 0.22, footwidth = 0.12,
                      ax=ax)
        # The shrinked footsize for defining contact points (smaller size)
        drawFootPatch(P=np.array([InitConfig["PRx_init"], InitConfig["PRy_init"], InitConfig["PRz_init"]]),
                      P_TangentX=InitConfig["PR_init_TangentX"], P_TangentY=InitConfig["PR_init_TangentY"], line_color=color, 
                      LineType = 'dashed', footlength = 0.2, footwidth = 0.1,
                      ax=ax)

    return ax

#   Draw Local Obj
def drawLocaObj(LocalObj=None, InitConfig=None, fig=None, ax=None, LineWidth=10):
    # draw CoM Obj
    ax.scatter(LocalObj["x_obj"],  LocalObj["y_obj"],
               LocalObj["z_obj"], c='c', marker='o', linewidth=LineWidth)
    # draw Contact location and foot patch
    ax.scatter(LocalObj["Px_obj"], LocalObj["Py_obj"],
               LocalObj["Pz_obj"], c='c', marker='o', linewidth=LineWidth)
    
    # The actual footsize (larger size)
    ax = drawFootPatch(P=np.concatenate(([LocalObj["Px_obj"], LocalObj["Py_obj"], LocalObj["Pz_obj"]]), axis=None),
                       P_TangentX=InitConfig["SurfTangentsX"][0], P_TangentY=InitConfig["SurfTangentsY"][0], line_color='c', 
                       LineType = 'solid', footlength = 0.22, footwidth = 0.12,
                       ax=ax)

    # The shrinked footsize for defining contact points (smaller size)
    ax = drawFootPatch(P=np.concatenate(([LocalObj["Px_obj"], LocalObj["Py_obj"], LocalObj["Pz_obj"]]), axis=None),
                       P_TangentX=InitConfig["SurfTangentsX"][0], P_TangentY=InitConfig["SurfTangentsY"][0], line_color='c', 
                       LineType = 'dashed', footlength = 0.2, footwidth = 0.1,
                       ax=ax)

    return ax

#   Draw Terminal Config
def drawTerminalConfig(TerminalConfig=None, InitConfig=None, fig=None, ax=None, color='k', LineWidth=5):
    # draw Terminal CoM
    ax.scatter(TerminalConfig["x_end"],  TerminalConfig["y_end"],
               TerminalConfig["z_end"], c=color, marker='o', linewidth=LineWidth)
    # draw Contact location and foot patch
    ax.scatter(TerminalConfig["Px"], TerminalConfig["Py"],
               TerminalConfig["Pz"], c=color, marker='o', linewidth=LineWidth)
    
    # The actual footsize (larger size)
    ax = drawFootPatch(P=np.concatenate(([TerminalConfig["Px"], TerminalConfig["Py"], TerminalConfig["Pz"]]), axis=None),
                       P_TangentX=InitConfig["SurfTangentsX"][0], P_TangentY=InitConfig["SurfTangentsY"][0], line_color=color, 
                       LineType = 'solid', footlength = 0.22, footwidth = 0.12, 
                       ax=ax)
    # The shrinked footsize for defining contact points (smaller size)
    ax = drawFootPatch(P=np.concatenate(([TerminalConfig["Px"], TerminalConfig["Py"], TerminalConfig["Pz"]]), axis=None),
                       P_TangentX=InitConfig["SurfTangentsX"][0], P_TangentY=InitConfig["SurfTangentsY"][0], line_color=color, 
                       LineType = 'dashed', footlength = 0.2, footwidth = 0.1, 
                       ax=ax)
    return ax

# Top Level Display Function
#   Operation Modes:
#       1) TerrainModel = Something, SingleOptResult = None,      AllOptResult = None:      Show Terrain Only
#       2) TerrainModel = Something, SingleOptResult = Something, AllOptResult = None:      Show Opimization Result of Current Round/Step (With the entire horizon)
#       3) TerrainModel = Something, SingleOptResult = None,      AllOptResult = Something: Show Optimization Result of All Rounds/Steps
def DisplayResults(TerrainModel=None, SingleOptResult=None, AllOptResult=None):
    # Open a draw space
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_ylim3d(-1, 1)

    # Plot According to Different Modes
    #   Draw Terrain Only
    if (TerrainModel != None) and (SingleOptResult == None) and (AllOptResult == None):
        # Draw Terrain
        print("Display Terrain Only")
        ax = drawTerrain(Sl0surf=TerrainModel["InitLeftSurfVertice"], Sr0surf=TerrainModel["InitRightSurfVertice"],
                         ContactSurfs=TerrainModel["ContactSurfsVertice"],
                         printTerrainVertice=True, fig=fig, ax=ax)
        # Label Terrain Patches
        ax = labelSurface(Sl0surf=TerrainModel["InitLeftSurfVertice"], Sr0surf=TerrainModel["InitRightSurfVertice"],
                          ContactSurfs=TerrainModel["ContactSurfsVertice"], fig=fig, ax=ax)
        ax.set_zlim([-0.01,1.0])
        ax.set_xlim([-1.0,3.0])
        plt.show()

    # Display Opimization Result of Current Round/Step (With the entire horizon)
    elif (TerrainModel != None) and (SingleOptResult != None) and (AllOptResult == None):
        print("Display Optimization Result of Current Round/Step")
        # Draw Terrain First
        ax = drawTerrain(Sl0surf=TerrainModel["InitLeftSurfVertice"], Sr0surf=TerrainModel["InitRightSurfVertice"],
                         ContactSurfs=TerrainModel["ContactSurfsVertice"],
                         printTerrainVertice=False, fig=fig, ax=ax)
        ax = drawSingleOptTraj(optResult=SingleOptResult, fig=fig, ax=ax)
        ax = labelSurface(Sl0surf=SingleOptResult["LeftInitSurf"], Sr0surf=SingleOptResult["RightInitSurf"],
                          ContactSurfs=SingleOptResult["ContactSurfs"], fig=fig, ax=ax)
        plt.show()

    # Display Optimization Result for all rounds/steps
    elif (TerrainModel != None) and (SingleOptResult == None) and (AllOptResult != None):
        if AllOptResult:
            print("Display Execution Horizon (First Step Only) for All Round/Step")
            ax = drawTerrain(Sl0surf=TerrainModel["InitLeftSurfVertice"], Sr0surf=TerrainModel["InitRightSurfVertice"],
                             ContactSurfs=TerrainModel["ContactSurfsVertice"],
                             printTerrainVertice=False, fig=fig, ax=ax)
            ax = DrawAllExecutionHorizon(
                allOptResult=AllOptResult, fig=fig, ax=ax)
            ax = labelSurface(Sl0surf=AllOptResult[0]["LeftInitSurf"], Sr0surf=AllOptResult[0]["RightInitSurf"],
                              ContactSurfs=TerrainModel["ContactSurfsVertice"], fig=fig, ax=ax)
            plt.show()

    # Show Figure
    # ax.set_box_aspect((1,1,1))

    return ax


def DisplayResults_Not_Show(TerrainModel=None, SingleOptResult=None, AllOptResult=None, fig=None, ax=None):
    # Open a draw space
    # fig=plt.figure();   ax = Axes3D(fig)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_ylim3d(-1, 1)

    # Plot According to Different Modes
    #   Draw Terrain Only
    if (TerrainModel != None) and (SingleOptResult == None) and (AllOptResult == None):
        # Draw Terrain
        print("Display Terrain Only")
        ax = drawTerrain(Sl0surf=TerrainModel["InitLeftSurfVertice"], Sr0surf=TerrainModel["InitRightSurfVertice"],
                         ContactSurfs=TerrainModel["ContactSurfsVertice"],
                         printTerrainVertice=True, fig=fig, ax=ax)
        # Label Terrain Patches
        ax = labelSurface(Sl0surf=TerrainModel["InitLeftSurfVertice"], Sr0surf=TerrainModel["InitRightSurfVertice"],
                          ContactSurfs=TerrainModel["ContactSurfsVertice"], fig=fig, ax=ax)

    # Display Opimization Result of Current Round/Step (With the entire horizon)
    elif (TerrainModel != None) and (SingleOptResult != None) and (AllOptResult == None):
        print("Display Optimization Result of Current Round/Step")
        # Draw Terrain First
        ax = drawTerrain(Sl0surf=TerrainModel["InitLeftSurfVertice"], Sr0surf=TerrainModel["InitRightSurfVertice"],
                         ContactSurfs=TerrainModel["ContactSurfsVertice"],
                         printTerrainVertice=False, fig=fig, ax=ax)
        ax = drawSingleOptTraj(optResult=SingleOptResult, fig=fig, ax=ax)
        ax = labelSurface(Sl0surf=SingleOptResult["LeftInitSurf"], Sr0surf=SingleOptResult["RightInitSurf"],
                          ContactSurfs=SingleOptResult["ContactSurfs"], fig=fig, ax=ax)

    # Display Optimization Result for all rounds/steps
    elif (TerrainModel != None) and (SingleOptResult == None) and (AllOptResult != None):
        if AllOptResult:
            print("Display Execution Horizon (First Step Only) for All Round/Step")
            ax = drawTerrain(Sl0surf=TerrainModel["InitLeftSurfVertice"], Sr0surf=TerrainModel["InitRightSurfVertice"],
                             ContactSurfs=TerrainModel["ContactSurfsVertice"],
                             printTerrainVertice=False, fig=fig, ax=ax)
            ax = DrawAllExecutionHorizon(
                allOptResult=AllOptResult, fig=fig, ax=ax)
            ax = labelSurface(Sl0surf=AllOptResult[0]["LeftInitSurf"], Sr0surf=AllOptResult[0]["RightInitSurf"],
                              ContactSurfs=TerrainModel["ContactSurfsVertice"], fig=fig, ax=ax)

    # Show Figure
    # ax.set_box_aspect((1,1,1))

    return ax

# Print Results of the Single Optimization


def PrintSingleOptTraj(SingleOptResult=None, LocalObjSettings=None, LocalObj=None):
    if SingleOptResult != None:  # print single Optimization result
        # Print First Level
        var_idx_lv1 = SingleOptResult["var_idx"]["Level1_Var_Index"]
        x_opt = SingleOptResult["opt_res"]

        print("Result of the First Level")
        x_res_lv1 = np.array(x_opt[var_idx_lv1["x"][0]:var_idx_lv1["x"][1]+1])
        print('x_res: ', x_res_lv1)
        y_res_lv1 = np.array(x_opt[var_idx_lv1["y"][0]:var_idx_lv1["y"][1]+1])
        print('y_res: ', y_res_lv1)
        z_res_lv1 = np.array(x_opt[var_idx_lv1["z"][0]:var_idx_lv1["z"][1]+1])
        print('z_res: ', z_res_lv1)
        xdot_res_lv1 = np.array(
            x_opt[var_idx_lv1["xdot"][0]:var_idx_lv1["xdot"][1]+1])
        print('xdot_res: ', xdot_res_lv1)
        ydot_res_lv1 = np.array(
            x_opt[var_idx_lv1["ydot"][0]:var_idx_lv1["ydot"][1]+1])
        print('ydot_res: ', ydot_res_lv1)
        zdot_res_lv1 = np.array(
            x_opt[var_idx_lv1["zdot"][0]:var_idx_lv1["zdot"][1]+1])
        print('zdot_res: ', zdot_res_lv1)
        Lx_res_lv1 = np.array(
            x_opt[var_idx_lv1["Lx"][0]:var_idx_lv1["Lx"][1]+1])
        print('Lx_res: ', Lx_res_lv1)
        Ly_res_lv1 = np.array(
            x_opt[var_idx_lv1["Ly"][0]:var_idx_lv1["Ly"][1]+1])
        print('Ly_res: ', Ly_res_lv1)
        Lz_res_lv1 = np.array(
            x_opt[var_idx_lv1["Lz"][0]:var_idx_lv1["Lz"][1]+1])
        print('Lz_res: ', Lz_res_lv1)
        Ldotx_res_lv1 = np.array(
            x_opt[var_idx_lv1["Ldotx"][0]:var_idx_lv1["Ldotx"][1]+1])
        print('Ldotx_res: ', Ldotx_res_lv1)
        Ldoty_res_lv1 = np.array(
            x_opt[var_idx_lv1["Ldoty"][0]:var_idx_lv1["Ldoty"][1]+1])
        print('Ldoty_res: ', Ldoty_res_lv1)
        Ldotz_res_lv1 = np.array(
            x_opt[var_idx_lv1["Ldotz"][0]:var_idx_lv1["Ldotz"][1]+1])
        print('Ldotz_res: ', Ldotz_res_lv1)
        px_res_lv1 = np.array(
            x_opt[var_idx_lv1["px"][0]:var_idx_lv1["px"][1]+1])
        print('px_res: ', px_res_lv1)
        py_res_lv1 = np.array(
            x_opt[var_idx_lv1["py"][0]:var_idx_lv1["py"][1]+1])
        print('py_res: ', py_res_lv1)
        pz_res_lv1 = np.array(
            x_opt[var_idx_lv1["pz"][0]:var_idx_lv1["pz"][1]+1])
        print('pz_res: ', pz_res_lv1)
        Ts_res_lv1 = np.array(
            x_opt[var_idx_lv1["Ts"][0]:var_idx_lv1["Ts"][1]+1])
        print('Ts_res: ', Ts_res_lv1)
        FR1x_res_lv1 = np.array(
            x_opt[var_idx_lv1["FR1x"][0]:var_idx_lv1["FR1x"][1]+1])
        print('FR1x_res: ', FR1x_res_lv1)
        FR1y_res_lv1 = np.array(
            x_opt[var_idx_lv1["FR1y"][0]:var_idx_lv1["FR1y"][1]+1])
        print('FR1y_res: ', FR1y_res_lv1)
        FR1z_res_lv1 = np.array(
            x_opt[var_idx_lv1["FR1z"][0]:var_idx_lv1["FR1z"][1]+1])
        print('FR1z_res: ', FR1z_res_lv1)

        # Print Local Obj Tracking Result As Well
        if LocalObjSettings["local_obj_tracking_type"] != None:
            print("-------------Local Obj Tracking Stats---------------")

            print("-----Terminal CoM Position--------")
            print("- Local Obj CoM Position: x = ", str(LocalObj["x_obj"]), ", y = ", str(
                LocalObj["y_obj"]), ", z = ", str(LocalObj["z_obj"]))
            print("- Planned CoM Position:   x = ",
                  str(x_res_lv1[-1]), ", y = ", str(y_res_lv1[-1]), ", z = ", str(z_res_lv1[-1]))
            print("- Local Obj and Plan CoM Diff (Abusolute Value): x = ", str(np.absolute(LocalObj["x_obj"]-x_res_lv1[-1])), "y = ", str(
                np.absolute(LocalObj["y_obj"]-y_res_lv1[-1])), "z = ", str(np.absolute(LocalObj["z_obj"]-z_res_lv1[-1])))

            print("-----Terminal CoM Velocity--------")
            print("- Local Obj CoM Velocity: xdot = ", str(LocalObj["xdot_obj"]), ", ydot = ", str(
                LocalObj["ydot_obj"]), ", zdot = ", str(LocalObj["zdot_obj"]))
            print("- Planned CoM Velocity:   xdot = ", str(xdot_res_lv1[-1]), ", ydot = ", str(
                ydot_res_lv1[-1]), ", zdot = ", str(zdot_res_lv1[-1]))
            print("- Local Obj and Plan CoM Diff (Abusolute Value): xdot = ", str(np.absolute(LocalObj["xdot_obj"]-xdot_res_lv1[-1])), "ydot = ", str(
                np.absolute(LocalObj["ydot_obj"]-ydot_res_lv1[-1])), "zdot = ", str(np.absolute(LocalObj["zdot_obj"]-zdot_res_lv1[-1])))

            print("-----Terminal Angular Momentum--------")
            print("- Local Obj Angular Momentum: Lx = ", str(LocalObj["Lx_obj"]), ", Ly = ", str(
                LocalObj["Ly_obj"]), ", Lz = ", str(LocalObj["Lz_obj"]))
            print("- Planned Angular Momentum:   Lx = ", str(Lx_res_lv1[-1]), ", Ly = ", str(
                ydot_res_lv1[-1]), ", Lz = ", str(Lz_res_lv1[-1]))
            print("- Local Obj and Plan Angular Momentum Diff (Abusolute Value): Lx = ", str(np.absolute(LocalObj["Lx_obj"]-Lx_res_lv1[-1])), "Ly = ", str(
                np.absolute(LocalObj["Ly_obj"]-Ly_res_lv1[-1])), "Lz = ", str(np.absolute(LocalObj["Lz_obj"]-Lz_res_lv1[-1])))

            print("-----Terminal Footstep Location--------")
            print("- Local Obj Footstep Location: Px = ", str(LocalObj["Px_obj"]), ", Py = ", str(
                LocalObj["Py_obj"]), ", Pz = ", str(LocalObj["Pz_obj"]))
            print("- Planned Footstep Location:   Px = ",
                  str(px_res_lv1[-1]), ", Py = ", str(py_res_lv1[-1]), ", Pz = ", str(pz_res_lv1[-1]))
            print("- Local Obj and Plan Footstep Location Diff (Abusolute Value): Px = ", str(np.absolute(LocalObj["Px_obj"]-px_res_lv1[-1])), "Py = ", str(
                np.absolute(LocalObj["Py_obj"]-py_res_lv1[-1])), "Pz = ", str(np.absolute(LocalObj["Pz_obj"]-pz_res_lv1[-1])))

        # Print level 2 result
        var_idx_lv2 = SingleOptResult["var_idx"]["Level2_Var_Index"]

        if var_idx_lv2:
            x_opt = x_opt[var_idx_lv1["Ts"][1]+1:]

            print("Result of the Second Level")
            print('x_res: ',     np.array(
                x_opt[var_idx_lv2["x"][0]:var_idx_lv2["x"][1]+1]))
            print('y_res: ',     np.array(
                x_opt[var_idx_lv2["y"][0]:var_idx_lv2["y"][1]+1]))
            print('z_res: ',     np.array(
                x_opt[var_idx_lv2["z"][0]:var_idx_lv2["z"][1]+1]))
            print('xdot_res: ',  np.array(
                x_opt[var_idx_lv2["xdot"][0]:var_idx_lv2["xdot"][1]+1]))
            print('ydot_res: ',  np.array(
                x_opt[var_idx_lv2["ydot"][0]:var_idx_lv2["ydot"][1]+1]))
            print('zdot_res: ',  np.array(
                x_opt[var_idx_lv2["zdot"][0]:var_idx_lv2["zdot"][1]+1]))
            print('Lx_res: ',    np.array(
                x_opt[var_idx_lv2["Lx"][0]:var_idx_lv2["Lx"][1]+1])) if "Lx" in var_idx_lv2.keys() else None
            print('Ly_res: ',    np.array(
                x_opt[var_idx_lv2["Ly"][0]:var_idx_lv2["Ly"][1]+1])) if "Ly" in var_idx_lv2.keys() else None
            print('Lz_res: ',    np.array(
                x_opt[var_idx_lv2["Lz"][0]:var_idx_lv2["Lz"][1]+1])) if "Lz" in var_idx_lv2.keys() else None
            print('Ldotx_res: ', np.array(
                x_opt[var_idx_lv2["Ldotx"][0]:var_idx_lv2["Ldotx"][1]+1])) if "Ldotx" in var_idx_lv2.keys() else None
            print('Ldoty_res: ', np.array(
                x_opt[var_idx_lv2["Ldoty"][0]:var_idx_lv2["Ldoty"][1]+1])) if "Ldoty" in var_idx_lv2.keys() else None
            print('Ldotz_res: ', np.array(
                x_opt[var_idx_lv2["Ldotz"][0]:var_idx_lv2["Ldotz"][1]+1])) if "Ldotz" in var_idx_lv2.keys() else None
            print('px_res: ',    np.array(
                x_opt[var_idx_lv2["px"][0]:var_idx_lv2["px"][1]+1]))
            print('py_res: ',    np.array(
                x_opt[var_idx_lv2["py"][0]:var_idx_lv2["py"][1]+1]))
            print('pz_res: ',    np.array(
                x_opt[var_idx_lv2["pz"][0]:var_idx_lv2["pz"][1]+1]))
            print('Ts_res: ',    np.array(
                x_opt[var_idx_lv2["Ts"][0]:var_idx_lv2["Ts"][1]+1]))
            print('FR1x_res: ',  np.array(
                x_opt[var_idx_lv2["FR1x"][0]:var_idx_lv2["FR1x"][1]+1])) if "FR1x" in var_idx_lv2.keys() else None
            print('FR1y_res: ',  np.array(
                x_opt[var_idx_lv2["FR1y"][0]:var_idx_lv2["FR1y"][1]+1])) if "FR1y" in var_idx_lv2.keys() else None
            print('FR1z_res: ',  np.array(
                x_opt[var_idx_lv2["FR1z"][0]:var_idx_lv2["FR1z"][1]+1])) if "FR1z" in var_idx_lv2.keys() else None

# Display Local Obj
#   Basic Mode: (Before Optimization)    Only Provide InitConfig and LocalObj, display what is the task
#   Advanced Mode: (After Optimization)  Display the Target LocalObj + Terminal Config after tracking + Ground Truth Terminal Config


def viewLocalObj(InitConfig=None, LocalObj=None, CurrentOptResult=None,
                 groundTruthTrajPath=None, roundNum=None, TerrainModel=None):

    fig = plt.figure()
    ax = Axes3D(fig)

    # Basic Mode: (Before Optimization) Only Provide InitConfig and LocalObj, and groundTruth display what is the task
    # Draw Terrain
    ax = drawTerrain(Sl0surf=TerrainModel["InitLeftSurfVertice"], Sr0surf=TerrainModel["InitRightSurfVertice"],
                     ContactSurfs=TerrainModel["ContactSurfsVertice"],
                     printTerrainVertice=False, fig=fig, ax=ax)
    # Draw Init Config for Current Round
    ax = drawInitConfig(InitConfig=InitConfig, fig=fig, ax=ax, LineWidth=10)
    # Draw Local Obj (From Ground Truth or Predicted)
    ax = drawLocaObj(LocalObj=LocalObj, InitConfig=InitConfig,
                     fig=fig, ax=ax, LineWidth=10)

    # Draw Ground Truth if we have
    if groundTruthTrajPath != None:
        with open(groundTruthTrajPath, 'rb') as f:
            data = pickle.load(f)

        groundTruth_InitConfig, groundTruth_TerminalState = getInitTerminalConfig(
            SingleOptRes=data["SingleOptResultSavings"][roundNum], Shift_World_Frame=None)

        ax = drawInitConfig(InitConfig=groundTruth_InitConfig,
                            fig=fig, ax=ax, color='k', LineWidth=5)
        ax = drawTerminalConfig(TerminalConfig=groundTruth_TerminalState,
                                InitConfig=groundTruth_InitConfig, fig=fig, ax=ax, color='k', LineWidth=5)

    # More advanced Mode, Draw Optimization Result
    if CurrentOptResult != None:
        # Get First Level Result
        var_idx_lv1 = CurrentOptResult["var_idx"]["Level1_Var_Index"]

        # Get Full optimization result
        x_opt = CurrentOptResult["opt_res"]

        # Get CoM res x, y, z
        x_lv1_res = np.array(x_opt[var_idx_lv1["x"][0]:var_idx_lv1["x"][1]+1])
        y_lv1_res = np.array(x_opt[var_idx_lv1["y"][0]:var_idx_lv1["y"][1]+1])
        z_lv1_res = np.array(x_opt[var_idx_lv1["z"][0]:var_idx_lv1["z"][1]+1])

        # Get Contact location
        px_lv1_res = np.array(
            x_opt[var_idx_lv1["px"][0]:var_idx_lv1["px"][1]+1])
        py_lv1_res = np.array(
            x_opt[var_idx_lv1["py"][0]:var_idx_lv1["py"][1]+1])
        pz_lv1_res = np.array(
            x_opt[var_idx_lv1["pz"][0]:var_idx_lv1["pz"][1]+1])

        # Plot CoM Traj
        ax.plot3D(x_lv1_res, y_lv1_res, z_lv1_res, color='b',
                  linestyle='dashed', linewidth=2, markersize=12)
        # Plot Contact Location
        if InitConfig["LeftSwingFlag"] == 1:
            ax.scatter(px_lv1_res, py_lv1_res, pz_lv1_res,
                       c='r', marker='o', linewidth=10)
        if InitConfig["RightSwingFlag"] == 1:
            ax.scatter(px_lv1_res, py_lv1_res, pz_lv1_res,
                       c='b', marker='o', linewidth=10)

        # Draw the Terminal (from Optimization) CoM
        ax.scatter(x_lv1_res[-1], y_lv1_res[-1],
                   z_lv1_res[-1], c='g', marker='o', linewidth=10)

        # Then Draw Ground Truth Traj, if we have
        if groundTruthTrajPath != None:
            with open(groundTruthTrajPath, 'rb') as f:
                data = pickle.load(f)

            x_opt = data["SingleOptResultSavings"][roundNum]["opt_res"]
            var_idx_lv1 = data["SingleOptResultSavings"][roundNum]["var_idx"]["Level1_Var_Index"]

            # Get CoM res x, y, z
            x_lv1_res = np.array(
                x_opt[var_idx_lv1["x"][0]:var_idx_lv1["x"][1]+1])
            y_lv1_res = np.array(
                x_opt[var_idx_lv1["y"][0]:var_idx_lv1["y"][1]+1])
            z_lv1_res = np.array(
                x_opt[var_idx_lv1["z"][0]:var_idx_lv1["z"][1]+1])

            # Get Contact location
            px_lv1_res = np.array(
                x_opt[var_idx_lv1["px"][0]:var_idx_lv1["px"][1]+1])
            py_lv1_res = np.array(
                x_opt[var_idx_lv1["py"][0]:var_idx_lv1["py"][1]+1])
            pz_lv1_res = np.array(
                x_opt[var_idx_lv1["pz"][0]:var_idx_lv1["pz"][1]+1])

            ax.plot3D(x_lv1_res, y_lv1_res, z_lv1_res, color='k',
                      linestyle='dashed', linewidth=2, markersize=12)

    ax.set_xlim3d(InitConfig["x_init"]-0.2, InitConfig["x_init"]+0.5)

    plt.show()
