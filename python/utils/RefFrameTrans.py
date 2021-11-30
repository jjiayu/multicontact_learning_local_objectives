#For Making Reference Frame Transformation

import numpy as np
import copy

#Shift World Frame to Local Init Contact Surf Frame
#Transformation Modes: 1) None 
#                      2) (Abandon) InitSurfBorder (left most border of the init contact patches, y = 0, z = 0)
#                      3) (Abandon) InitCoM (but y = 0, z = 0)
#                      4) StanceFoot (Most Useful): represent quantities in the local contact frame of the stance foot
#NOTE: the function copies the InitConfig we pass into, and Tangent, Norm Orientaiton are NOT Transformed (they are not used in the formulation)
def shiftInitTerminalConfig_to_LocalFrame(InitConfig = None, TerminalConfig = None, Local_Frame_Selection = None):
    if InitConfig == None:
        raise Exception("Init Config is not provided")

    #----------------------------------------------------------------------------------------
    #Copy to get new instances for (shifted) Init and Terminal Configs
    shiftedInitConfig = copy.deepcopy(InitConfig);     shiftedTerminalConfig = copy.deepcopy(TerminalConfig)
    
    #----------------------------------------------------------------------------------------
    #Get Transformation (from local frame in the world frame)
    RotateTran, HomoTran = Compute_Local_to_World_Frame_Transformation(InitConfig = InitConfig, Local_Frame_Name = Local_Frame_Selection)

    #----------------------------------------------------------------------------------------
    #Transform InitConfig Quantities (InitConfig and TerminalConfig Always in World Frame, we need to tranform to local frame, with an inverse)
    #   InitConfig CoM x, y, z, Apply HomoTran
    initCoM_aug = np.array([[shiftedInitConfig["x_init"],shiftedInitConfig["y_init"],shiftedInitConfig["z_init"],1.]]).T #augmented Init CoM vector
    initCoM_aug_local = np.linalg.inv(HomoTran)@initCoM_aug #get liniocal representation by applying the homogeneous transformation
    
    shiftedInitConfig["x_init"] = initCoM_aug_local[0][0]; shiftedInitConfig["y_init"] = initCoM_aug_local[1][0] 
    shiftedInitConfig["z_init"] = initCoM_aug_local[2][0]

    #    InitConfig CoMdot x, y, z, Apply Rotation Tran
    initCoMdot_aug = np.array([[shiftedInitConfig["xdot_init"],shiftedInitConfig["ydot_init"],shiftedInitConfig["zdot_init"],1.]]).T #augmented Init CoMdot vector
    initCoMdot_aug_local = np.linalg.inv(RotateTran)@initCoMdot_aug

    shiftedInitConfig["xdot_init"] = initCoMdot_aug_local[0][0]; shiftedInitConfig["ydot_init"] = initCoMdot_aug_local[1][0]
    shiftedInitConfig["zdot_init"] = initCoMdot_aug_local[2][0]
    
    #   Init Angular Momentum, Apply Rotation Tran
    initL_aug = np.array([[shiftedInitConfig["Lx_init"],shiftedInitConfig["Ly_init"],shiftedInitConfig["Lz_init"],1.]]).T #augmented Init CoMdot vector
    initL_aug_local = np.linalg.inv(RotateTran)@initL_aug

    shiftedInitConfig["Lx_init"] = initL_aug_local[0][0]; shiftedInitConfig["Ly_init"] = initL_aug_local[1][0]
    shiftedInitConfig["Lz_init"] = initL_aug_local[2][0]

    #   Init Angular Momentum Rate (Skip, Not Important)
    
    #   Init Contact Locations, Appply HomoTran
    #   Left Init Contact Location
    initPL_aug = np.array([[shiftedInitConfig["PLx_init"],shiftedInitConfig["PLy_init"],shiftedInitConfig["PLz_init"],1.]]).T
    initPL_aug_local = np.linalg.inv(HomoTran)@initPL_aug

    shiftedInitConfig["PLx_init"] = initPL_aug_local[0][0]; shiftedInitConfig["PLy_init"] = initPL_aug_local[1][0]
    shiftedInitConfig["PLz_init"] = initPL_aug_local[2][0]

    #   Right Init Contact Location
    initPR_aug = np.array([[shiftedInitConfig["PRx_init"],shiftedInitConfig["PRy_init"],shiftedInitConfig["PRz_init"],1.]]).T
    initPR_aug_local = np.linalg.inv(HomoTran)@initPR_aug

    shiftedInitConfig["PRx_init"] = initPR_aug_local[0][0]; shiftedInitConfig["PRy_init"] = initPR_aug_local[1][0]
    shiftedInitConfig["PRz_init"] = initPR_aug_local[2][0]

    #Init Contact Surfaces, Apply Homo Tran
    #   LEFT Init Contact Patch
    #   NOTE: For the followings, we need to take the entire row (dim 2), so the index is like starting_row:starting_row + 1
    LeftInitPatch_Vertex1_aug = np.hstack((shiftedInitConfig["LeftInitSurf"][0:1],np.array([[1.]]))).T
    LeftInitPatch_Vertex1_aug_local = np.linalg.inv(HomoTran)@LeftInitPatch_Vertex1_aug

    LeftInitPatch_Vertex2_aug = np.hstack((shiftedInitConfig["LeftInitSurf"][1:2],np.array([[1.]]))).T
    LeftInitPatch_Vertex2_aug_local = np.linalg.inv(HomoTran)@LeftInitPatch_Vertex2_aug

    LeftInitPatch_Vertex3_aug = np.hstack((shiftedInitConfig["LeftInitSurf"][2:3],np.array([[1.]]))).T
    LeftInitPatch_Vertex3_aug_local = np.linalg.inv(HomoTran)@LeftInitPatch_Vertex3_aug

    LeftInitPatch_Vertex4_aug = np.hstack((shiftedInitConfig["LeftInitSurf"][3:],np.array([[1.]]))).T
    LeftInitPatch_Vertex4_aug_local = np.linalg.inv(HomoTran)@LeftInitPatch_Vertex4_aug
    #   Rebuild Contact Patch
    shiftedInitConfig["LeftInitSurf"] = np.vstack((LeftInitPatch_Vertex1_aug_local[0:3].T, LeftInitPatch_Vertex2_aug_local[0:3].T,
                                                   LeftInitPatch_Vertex3_aug_local[0:3].T, LeftInitPatch_Vertex4_aug_local[0:3].T))

    #   Right Init Contact Patch
    RightInitPatch_Vertex1_aug = np.hstack((shiftedInitConfig["RightInitSurf"][0:1],np.array([[1.]]))).T
    RightInitPatch_Vertex1_aug_local = np.linalg.inv(HomoTran)@RightInitPatch_Vertex1_aug

    RightInitPatch_Vertex2_aug = np.hstack((shiftedInitConfig["RightInitSurf"][1:2],np.array([[1.]]))).T
    RightInitPatch_Vertex2_aug_local = np.linalg.inv(HomoTran)@RightInitPatch_Vertex2_aug

    RightInitPatch_Vertex3_aug = np.hstack((shiftedInitConfig["RightInitSurf"][2:3],np.array([[1.]]))).T
    RightInitPatch_Vertex3_aug_local = np.linalg.inv(HomoTran)@RightInitPatch_Vertex3_aug

    RightInitPatch_Vertex4_aug = np.hstack((shiftedInitConfig["RightInitSurf"][3:],np.array([[1.]]))).T
    RightInitPatch_Vertex4_aug_local = np.linalg.inv(HomoTran)@RightInitPatch_Vertex4_aug

    #   Rebuild Contact Patch
    shiftedInitConfig["RightInitSurf"] = np.vstack((RightInitPatch_Vertex1_aug_local[0:3].T, RightInitPatch_Vertex2_aug_local[0:3].T,
                                                    RightInitPatch_Vertex3_aug_local[0:3].T, RightInitPatch_Vertex4_aug_local[0:3].T))

    #Shift Contact Surfaces, Apply HomoTran
    for cont_surf_cnt in range(len(shiftedInitConfig["ContactSurfs"])):
        contactsurf_vertex1_aug = np.hstack((shiftedInitConfig["ContactSurfs"][cont_surf_cnt][0:1],np.array([[1.]]))).T
        contactsurf_vertex1_aug_local = np.linalg.inv(HomoTran)@contactsurf_vertex1_aug

        contactsurf_vertex2_aug = np.hstack((shiftedInitConfig["ContactSurfs"][cont_surf_cnt][1:2],np.array([[1.]]))).T
        contactsurf_vertex2_aug_local = np.linalg.inv(HomoTran)@contactsurf_vertex2_aug

        contactsurf_vertex3_aug = np.hstack((shiftedInitConfig["ContactSurfs"][cont_surf_cnt][2:3],np.array([[1.]]))).T
        contactsurf_vertex3_aug_local = np.linalg.inv(HomoTran)@contactsurf_vertex3_aug

        contactsurf_vertex4_aug = np.hstack((shiftedInitConfig["ContactSurfs"][cont_surf_cnt][3:],np.array([[1.]]))).T
        contactsurf_vertex4_aug_local = np.linalg.inv(HomoTran)@contactsurf_vertex4_aug

        shiftedInitConfig["ContactSurfs"][cont_surf_cnt] = np.vstack((contactsurf_vertex1_aug_local[0:3].T, contactsurf_vertex2_aug_local[0:3].T,
                                                                      contactsurf_vertex3_aug_local[0:3].T, contactsurf_vertex4_aug_local[0:3].T))

    #For moving prediction patches, Apply HomoTran
    if "PreviewSurfs_for_Prediction" in shiftedInitConfig:
        for cont_surf_cnt in range(len(shiftedInitConfig["PreviewSurfs_for_Prediction"])):
            previewsurf_vertex1_aug = np.hstack((shiftedInitConfig["PreviewSurfs_for_Prediction"][cont_surf_cnt][0:1],np.array([[1.]]))).T
            previewsurf_vertex1_aug_local = np.linalg.inv(HomoTran)@previewsurf_vertex1_aug

            previewsurf_vertex2_aug = np.hstack((shiftedInitConfig["PreviewSurfs_for_Prediction"][cont_surf_cnt][1:2],np.array([[1.]]))).T
            previewsurf_vertex2_aug_local = np.linalg.inv(HomoTran)@previewsurf_vertex2_aug

            previewsurf_vertex3_aug = np.hstack((shiftedInitConfig["PreviewSurfs_for_Prediction"][cont_surf_cnt][2:3],np.array([[1.]]))).T
            previewsurf_vertex3_aug_local = np.linalg.inv(HomoTran)@previewsurf_vertex3_aug

            previewsurf_vertex4_aug = np.hstack((shiftedInitConfig["PreviewSurfs_for_Prediction"][cont_surf_cnt][3:],np.array([[1.]]))).T
            previewsurf_vertex4_aug_local = np.linalg.inv(HomoTran)@previewsurf_vertex4_aug

            shiftedInitConfig["PreviewSurfs_for_Prediction"][cont_surf_cnt] = np.vstack((previewsurf_vertex1_aug_local[0:3].T, previewsurf_vertex2_aug_local[0:3].T,
                                                                                         previewsurf_vertex3_aug_local[0:3].T, previewsurf_vertex4_aug_local[0:3].T))
    
    #----------------------------------------------------------------------------------------
    #For Terminal States
    if TerminalConfig != None:
        #Terminal CoM x, y, z, Apply HomoTran
        terminalCoM_aug = np.array([[shiftedTerminalConfig["x_end"],shiftedTerminalConfig["y_end"],shiftedTerminalConfig["z_end"],1.]]).T #augmented Init CoM vector
        terminalCoM_aug_local = np.linalg.inv(HomoTran)@terminalCoM_aug #get liniocal representation by applying the homogeneous transformation
        
        shiftedTerminalConfig["x_end"] = terminalCoM_aug_local[0][0]; shiftedTerminalConfig["y_end"] = terminalCoM_aug_local[1][0] 
        shiftedTerminalConfig["z_end"] = terminalCoM_aug_local[2][0]

        #    TerminalConfig CoMdot x, y, z, Apply Rotation Tran
        terminalCoMdot_aug = np.array([[shiftedTerminalConfig["xdot_end"],shiftedTerminalConfig["ydot_end"],shiftedTerminalConfig["zdot_end"],1.]]).T #augmented Init CoMdot vector
        terminalCoMdot_aug_local = np.linalg.inv(RotateTran)@terminalCoMdot_aug

        shiftedTerminalConfig["xdot_end"] = terminalCoMdot_aug_local[0][0]; shiftedTerminalConfig["ydot_end"] = terminalCoMdot_aug_local[1][0]
        shiftedTerminalConfig["zdot_end"] = terminalCoMdot_aug_local[2][0]
        
        #   TerminalConfig Angular Momentum, Apply Rotation Tran
        terminalL_aug = np.array([[shiftedTerminalConfig["Lx_end"],shiftedTerminalConfig["Ly_end"],shiftedTerminalConfig["Lz_end"],1.]]).T #augmented Init CoMdot vector
        terminalL_aug_local = np.linalg.inv(RotateTran)@terminalL_aug

        shiftedTerminalConfig["Lx_end"] = terminalL_aug_local[0][0]; shiftedTerminalConfig["Ly_end"] = terminalL_aug_local[1][0]
        shiftedTerminalConfig["Lz_end"] = terminalL_aug_local[2][0]

        #Terminal Contact Location, Apply Homo Tran
        terminalP_aug = np.array([[shiftedTerminalConfig["Px"],shiftedTerminalConfig["Py"],shiftedTerminalConfig["Pz"],1.]]).T
        terminalP_aug_local = np.linalg.inv(HomoTran)@terminalP_aug

        shiftedTerminalConfig["Px"] = terminalP_aug_local[0][0]; shiftedTerminalConfig["Py"] = terminalP_aug_local[1][0]
        shiftedTerminalConfig["Pz"] = terminalP_aug_local[2][0]

    return shiftedInitConfig, shiftedTerminalConfig

#Shift from Local frame to World Frame
def shiftLocalObj_to_WorldFrame(InitConfig = None, LocalObj = None, Local_Frame_Selection = None):
    #-----------------------------------------------
    #Make Result Container
    shiftedLocalObj = copy.deepcopy(LocalObj)

    #-----------------------
    #Get Transformation (from local frame in the world frame)
    RotateTran, HomoTran = Compute_Local_to_World_Frame_Transformation(InitConfig = InitConfig, Local_Frame_Name = Local_Frame_Selection)

    #-------------------------------------------------------
    #Transform Variables (Local Obj always in Local Frame we need to convert to World Frame)
    #   Local Obj CoM x, y, z, Apply Homotran
    localCoM_aug = np.array([[shiftedLocalObj["x_obj"],shiftedLocalObj["y_obj"],shiftedLocalObj["z_obj"],1.]]).T #augmented Init CoM vector
    localCoM_aug_local = HomoTran@localCoM_aug #get liniocal representation by applying the homogeneous transformation
    
    shiftedLocalObj["x_obj"] = localCoM_aug_local[0][0]; shiftedLocalObj["y_obj"] = localCoM_aug_local[1][0] 
    shiftedLocalObj["z_obj"] = localCoM_aug_local[2][0]

    #    Local Obj CoMdot x, y, z, Apply Rotation Tran
    localCoMdot_aug = np.array([[shiftedLocalObj["xdot_obj"],shiftedLocalObj["ydot_obj"],shiftedLocalObj["zdot_obj"],1.]]).T #augmented Init CoMdot vector
    localCoMdot_aug_local = RotateTran@localCoMdot_aug

    shiftedLocalObj["xdot_obj"] = localCoMdot_aug_local[0][0]; shiftedLocalObj["ydot_obj"] = localCoMdot_aug_local[1][0]
    shiftedLocalObj["zdot_obj"] = localCoMdot_aug_local[2][0]

    #   Local Obj Angular Momentum, Apply Rotation Tran
    localL_aug = np.array([[shiftedLocalObj["Lx_obj"],shiftedLocalObj["Ly_obj"],shiftedLocalObj["Lz_obj"],1.]]).T #augmented Init CoMdot vector
    localL_aug_local = RotateTran@localL_aug

    shiftedLocalObj["Lx_obj"] = localL_aug_local[0][0]; shiftedLocalObj["Ly_obj"] = localL_aug_local[1][0]
    shiftedLocalObj["Lz_obj"] = localL_aug_local[2][0]

    #   Local Obj Contact Location Target, Apply HomoTran
    localP_aug = np.array([[shiftedLocalObj["Px_obj"],shiftedLocalObj["Py_obj"],shiftedLocalObj["Pz_obj"],1.]]).T #augmented Init CoMdot vector
    localP_aug_local = HomoTran@localP_aug

    shiftedLocalObj["Px_obj"] = localP_aug_local[0][0]; shiftedLocalObj["Py_obj"] = localP_aug_local[1][0]
    shiftedLocalObj["Pz_obj"] = localP_aug_local[2][0]

    return shiftedLocalObj

#NOTE: InitConfig has to be UnShifted Ones, (Basically In the World Frame)
def Compute_Local_to_World_Frame_Transformation(InitConfig = None, Local_Frame_Name = None):
    #----------------------------------------------------------------------------------------
    #Build Homogeneouse Transformation based on Transformation mode
    #   Define Non-moving case
    Id_RotateMatrix = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
    ZeroTrans_Vec   = np.array([[0.,0.,0.]]).T

    #----------------------------------------------------------------------------------------
    #   Get Rotation Matrix and Translation Vector
    if Local_Frame_Name == None:#No shift
        rotation_matrix = Id_RotateMatrix;     translation_vec = ZeroTrans_Vec
    #elif Local_Frame_Name == "InitSurfBorder": #Shift to the left most border of the init contact patches
    #    #Surf Vertex Identification
    #    #p2---------------------p1
    #    # |                      |
    #    # |                      |
    #    # |                      |
    #    #p3---------------------p4
    #    #we only move the x-axis, so we only manipulate the right top corner
    #    rotation_matrix = Id_RotateMatrix;      translation_vec = ZeroTrans_Vec
    #    translation_vec[0] = np.array(np.min([InitConfig["LeftInitSurf"][1][0], InitConfig["RightInitSurf"][1][0]]))
    #elif Local_Frame_Name == "InitCoM": #Shift to the Init CoM Frame (NOTE:Only Shifted x-axis)
    #    rotation_matrix = Id_RotateMatrix;      translation_vec = ZeroTrans_Vec
    #    translation_vec[0] = InitConfig["x_init"]
    elif Local_Frame_Name == "StanceFoot":
        #Decide Contact Location and Orientation for the stance foot
        if   (InitConfig["LeftSwingFlag"] == 1) and (InitConfig["RightSwingFlag"] == 0): #Swing the Left foot, then the Right Foot is the STANCE Foot
            rotation_matrix = InitConfig["RightInitSurfOrientation"] #Contact Orientation
            translation_vec = np.array([[InitConfig["PRx_init"],InitConfig["PRy_init"],InitConfig["PRz_init"]]]).T #Contact Location Vector, Define as a column vector
        elif (InitConfig["LeftSwingFlag"] == 0) and (InitConfig["RightSwingFlag"] == 1): #Swing the Right foot, then the Left Foot is the STANCE Foot
            rotation_matrix = InitConfig["LeftInitSurfOrientation"]  #Contact Orientation
            translation_vec = np.array([[InitConfig["PLx_init"],InitConfig["PLy_init"],InitConfig["PLz_init"]]]).T #Contact Location Vector, Define as a column vector
    else:
        raise Exception("Unknow Local_Frame_Selection Model in Function shiftInitConfig_to_LocalFrame")

    #   Build Homogeneous Transformation Matrix
    #   Homo Transformation with only Rotation Matrix (4x4)
    RotateTran = np.hstack((rotation_matrix, ZeroTrans_Vec))
    RotateTran = np.vstack((RotateTran, np.array([[0.,0.,0.,1.]])))
    #   Homo Transformation with Rotation Matrix and translation
    HomoTran = np.hstack((rotation_matrix, translation_vec))
    HomoTran = np.vstack((HomoTran,np.array([[0.,0.,0.,1.]])))

    return RotateTran, HomoTran