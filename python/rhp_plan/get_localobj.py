import numpy as np
import pickle
from multicontact_learning_local_objectives.python.utils import *
import os
import copy


# Get Local obj (unshifted, quantities can be in the local frame, in rhp_gen, the transformation to global frame is processed)
#   Switch = None: we dont have local obj, but we need to fill some random number to casadi parameters
#   Switch = fromFile: get local obj from file, need input refTrajFile defines the traj for tracking, shift_world_frame: shift coordinate to local frames,
#   roundNum: tell which round we are querying
#   ContactParameterizationType: tells how we parameterize the contact location:
#                                1) 3DPoints; 2) ConvexCombination (of all vertices); 3) FollowRectangelBorder
def getLocalobj(Mode=None, refTrajFile=None, roundNum=None,
                MLModelPath=None, shift_world_frame=None, ScaleFactor=1,
                ContactParameterizationType=None,
                InitConfig=None,
                DataSetPath=None):

    # Make Local Obj Container
    localobj = {}

    # Get From File Directly
    if Mode == "fromFile":

        print(" ")
        print("Get Local Obj from File, with Timing Info")
        print(" ")

        with open(refTrajFile, 'rb') as f:
            data = pickle.load(f)

        # Get result for the pre-specified roundNum
        optResult = data["SingleOptResultSavings"][roundNum]

        # Get InitConfig and Terminal Config with Scaled Factor
        InitConfig, TerminalConfig = getInitTerminalConfig(
            SingleOptRes=optResult, Shift_World_Frame=shift_world_frame)

        # Span the variables, NOTE:Remember to Scale things Back
        localobj["x_obj"] = TerminalConfig["x_end"]
        localobj["y_obj"] = TerminalConfig["y_end"]
        localobj["z_obj"] = TerminalConfig["z_end"]

        localobj["xdot_obj"] = TerminalConfig["xdot_end"]
        localobj["ydot_obj"] = TerminalConfig["ydot_end"]
        localobj["zdot_obj"] = TerminalConfig["zdot_end"]

        localobj["Lx_obj"] = TerminalConfig["Lx_end"]
        localobj["Ly_obj"] = TerminalConfig["Ly_end"]
        localobj["Lz_obj"] = TerminalConfig["Lz_end"]

        # Get Phase Duration
        var_Idx_lv1 = optResult["var_idx"]["Level1_Var_Index"]
        opt_res = optResult["opt_res"]

        Ts_vec = opt_res[var_Idx_lv1["Ts"][0]:var_Idx_lv1["Ts"][1]+1]
        print("Timing Vec", Ts_vec)

        InitDS_Dur = Ts_vec[-3]
        SS_Dur = Ts_vec[-2] - Ts_vec[-3]
        DS_Dur = Ts_vec[-1] - Ts_vec[-2]

        localobj["InitDS_Ts_obj"] = 0 + InitDS_Dur
        localobj["SS_Ts_obj"] = InitDS_Dur + SS_Dur
        localobj["DS_Ts_obj"] = InitDS_Dur + SS_Dur + DS_Dur

        # Naive Option: 3D point representation
        if ContactParameterizationType == "3DPoints":
            localobj["Px_obj"] = TerminalConfig["Px"]
            localobj["Py_obj"] = TerminalConfig["Py"]
            localobj["Pz_obj"] = TerminalConfig["Pz"]

        elif ContactParameterizationType == "ConvexCombination":
            # Generate Contact Location List
            P_land_ref = np.concatenate(
                (TerminalConfig["Px"], TerminalConfig["Py"], TerminalConfig["Pz"]), axis=None)
            # Get the first Patch, and place vertex in a column vector fashion [v1,v2,v3,v4]
            FirstPatch = InitConfig["ContactSurfs"][0]
            # Compute Coefficient for convex combination
            coefs = Point3D_to_ConvexCombination(
                ContactLocation=P_land_ref, ContactSurf=FirstPatch)
            # Recover to Local Obj
            P_localobj = ConvexCombination_to_Point3D(
                Coef=coefs, ContactSurf=FirstPatch)
            # Convert to Local Obj
            localobj["Px_obj"] = P_localobj[0]
            localobj["Py_obj"] = P_localobj[1]
            localobj["Pz_obj"] = P_localobj[2]

        elif ContactParameterizationType == "FollowRectangelBorder":
            P_land_ref = np.array(
                [TerminalConfig["Px"], TerminalConfig["Py"], TerminalConfig["Pz"]])  # A vertical vector
            # Contact Surface Placed Row by Row
            FirstPatch = InitConfig["ContactSurfs"][0]
            # 3x3 symmetric matrix
            FirstPatchOrientation = InitConfig["SurfOrientations"][0]
            # The third point should be the local origin (index 2); NOTE: A row vector now, but in two dimension
            FirstPatchLocalOrigin = np.array([FirstPatch[2]])
            # Make Homogeneous Transformation
            HomoTran = np.hstack(
                (FirstPatchOrientation, FirstPatchLocalOrigin.T))
            HomoTran = np.vstack((HomoTran, np.array([[0.0, 0.0, 0.0, 1.0]])))
            # Transform Quantities in Local Frame (all quantities should have 0 z-axis, as all stay in the same plane and the origin should be 0,0,0)
            #   Contact Location (augumented with 1 for homotran)
            P_land_local_aug = np.linalg.inv(HomoTran)@np.vstack((P_land_ref, np.array([[1]])))
            #   Vertex 0-3 (augumented with 1 for homotran)
            Vertex0_local_aug = np.linalg.inv(HomoTran)@np.vstack((np.array([FirstPatch[0]]).T, np.array([[1]])))
            Vertex1_local_aug = np.linalg.inv(HomoTran)@np.vstack((np.array([FirstPatch[1]]).T, np.array([[1]])))
            Vertex2_local_aug = np.linalg.inv(HomoTran)@np.vstack((np.array([FirstPatch[2]]).T, np.array([[1]])))
            Vertex3_local_aug = np.linalg.inv(HomoTran)@np.vstack((np.array([FirstPatch[3]]).T, np.array([[1]])))
            # Get 2D Vectors
            P_land_local_2d = P_land_local_aug[0:2]
            Vertex0_local_2d = Vertex0_local_aug[0:2]
            Vertex1_local_2d = Vertex1_local_aug[0:2]
            Vertex2_local_2d = Vertex2_local_aug[0:2]
            Vertex3_local_2d = Vertex3_local_aug[0:2]
            # Get local axis
            #   x axis: First Vertex (index 0)- Second Vertex (index 1)
            local_x_vec = Vertex0_local_2d - Vertex1_local_2d
            #   y axis: Third Vertex (index 1) - Second Vertex (index 2)
            local_y_vec = Vertex1_local_2d - Vertex2_local_2d
            # Build Matrix and Compute Moving Extent
            TransMatrix = np.hstack((local_x_vec, local_y_vec))
            coefs = np.linalg.inv(TransMatrix)@P_land_local_2d
            # Convert back to 3D contact location
            #   x and y vector for the terrain in world frame (i.e. original world frame, shifted world frame)
            terrain_x_vec = np.array(
                [FirstPatch[0]]).T - np.array([FirstPatch[1]]).T
            terrain_y_vec = np.array(
                [FirstPatch[1]]).T - np.array([FirstPatch[2]]).T
            # Get P_land (Need to add origin and then add the vertial and horizontal displacement)
            P_localobj = np.array([FirstPatch[2]]).T + np.hstack((terrain_x_vec, terrain_y_vec))@coefs
            localobj["Px_obj"] = P_localobj[0]
            localobj["Py_obj"] = P_localobj[1]
            localobj["Pz_obj"] = P_localobj[2]
        else:
            raise Exception(
                "Undefined Contact Location Parameterization Type (ContactParameterizationType)")

    elif Mode == "NeuralNetwork":  # NOTE: Everything in Local Frame

        print("--------------")
        print("Get Local Obj from Neural Network")
        print("--------------")

        # --------
        # TensorFlow when use as tracking local obj
        # Use CPU for Tensorflow
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        # Load NN Model
        NNmodel = load_model(MLModelPath, compile=False)
        # Load DataSet Settings
        DataSetSettingsPath = MLModelPath + "/" + "datasetSettings.p"
        DataSetSettings = pickle.load(open(DataSetSettingsPath, "rb"))

        # Build Input Vector (Need to Shift Vectors according to shift_world_frame_mode)
        shiftedInitConfig, shiftedTerminalConfig = shiftInitTerminalConfig_to_LocalFrame(
            InitConfig=InitConfig, TerminalConfig=None, Local_Frame_Selection=shift_world_frame)

        if shift_world_frame == None:  # No transformation to local frames, then full input vector
            InputVector = (shiftedInitConfig["x_init"],      shiftedInitConfig["y_init"],      shiftedInitConfig["z_init"],
                           shiftedInitConfig["xdot_init"],   shiftedInitConfig["ydot_init"],   shiftedInitConfig["zdot_init"],
                           shiftedInitConfig["Lx_init"],     shiftedInitConfig["Ly_init"],     shiftedInitConfig["Lz_init"],
                           shiftedInitConfig["PLx_init"],    shiftedInitConfig["PLy_init"],    shiftedInitConfig["PLz_init"],
                           shiftedInitConfig["PRx_init"],    shiftedInitConfig["PRy_init"],    shiftedInitConfig["PRz_init"],
                           shiftedInitConfig["LeftSwingFlag"],
                           shiftedInitConfig["LeftInitSurf"], shiftedInitConfig["RightInitSurf"],
                           shiftedInitConfig["PreviewSurfs_for_Prediction"])
        elif shift_world_frame == "StanceFoot":
            # Decide Contact Location before Swing, NOTE: when in StanceFoot frame, the stationary foot location is always 0, so we ignore that
            # Swing Left Foot
            if (shiftedInitConfig["LeftSwingFlag"] == 1) and (shiftedInitConfig["RightSwingFlag"] == 0):
                contact_before_swing_X = shiftedInitConfig["PLx_init"]
                contact_before_swing_Y = shiftedInitConfig["PLy_init"]
                contact_before_swing_Z = shiftedInitConfig["PLz_init"]
            # Swing Right Foot
            elif (shiftedInitConfig["LeftSwingFlag"] == 0) and (shiftedInitConfig["RightSwingFlag"] == 1):
                contact_before_swing_X = shiftedInitConfig["PRx_init"]
                contact_before_swing_Y = shiftedInitConfig["PRy_init"]
                contact_before_swing_Z = shiftedInitConfig["PRz_init"]
            else:
                raise Exception("Unknow Leg Swing Indicators")
            # Collect Input Vector
            InputVector = (shiftedInitConfig["x_init"],      shiftedInitConfig["y_init"],      shiftedInitConfig["z_init"],
                           shiftedInitConfig["xdot_init"],   shiftedInitConfig["ydot_init"],   shiftedInitConfig["zdot_init"],
                           shiftedInitConfig["Lx_init"],     shiftedInitConfig["Ly_init"],     shiftedInitConfig["Lz_init"],
                           contact_before_swing_X,           contact_before_swing_Y,           contact_before_swing_Z,
                           shiftedInitConfig["LeftSwingFlag"],
                           shiftedInitConfig["LeftInitSurf"], shiftedInitConfig["RightInitSurf"],
                           shiftedInitConfig["PreviewSurfs_for_Prediction"])
        else:
            raise Exception("Unknown Mode of Shifting to Local Frame")

        # Put input into Vector form
        InputVector = np.array([np.concatenate(InputVector, axis=None)])

        # NOTE: Scale the input vector according to the type of the scaling
        if DataSetSettings["PreProcessMode"] == "Standarization" or DataSetSettings["PreProcessMode"] == "MaxAbs":
            print("Transfrom Input from Original Form into: ",
                  DataSetSettings["PreProcessMode"])
            InputVector = DataSetSettings["Scaler_X"].transform(InputVector)
        elif DataSetSettings["PreProcessMode"] == "OriginalForm":
            print("Data Represented in Original Form, Scale with a constant")
            InputVector = InputVector*DataSetSettings["VectorScaleFactor"]
        else:
            print("Unknown Pre Processing Model")

        # Predict the output
        PredictVector = NNmodel.predict(InputVector)

        # Scale the Output Back
        if DataSetSettings["PreProcessMode"] == "Standarization" or DataSetSettings["PreProcessMode"] == "MaxAbs":
            print("Transfrom Output from: ",
                  DataSetSettings["PreProcessMode"], ", back to Original Form")
            PredictVector = DataSetSettings["Scaler_Y"].inverse_transform(
                PredictVector)
        elif DataSetSettings["PreProcessMode"] == "OriginalForm":
            print("Data Represented in Original Form, Scale with a constant")
            PredictVector = PredictVector/DataSetSettings["VectorScaleFactor"]
        else:
            print("Unknown Pre Processing Model")

        # Make Local Obj Dict (NOTE: the PredictVector is a 2D array)
        localobj["x_obj"] = PredictVector[0][0]
        localobj["y_obj"] = PredictVector[0][1]
        localobj["z_obj"] = PredictVector[0][2]
        localobj["xdot_obj"] = PredictVector[0][3]
        localobj["ydot_obj"] = PredictVector[0][4]
        localobj["zdot_obj"] = PredictVector[0][5]
        localobj["Lx_obj"] = PredictVector[0][6]
        localobj["Ly_obj"] = PredictVector[0][7]
        localobj["Lz_obj"] = PredictVector[0][8]

        #   Make Contact location
        if ContactParameterizationType == "3DPoints":
            localobj["Px_obj"] = PredictVector[0][9]
            localobj["Py_obj"] = PredictVector[0][10]
            localobj["Pz_obj"] = PredictVector[0][11]
        elif ContactParameterizationType == "ConvexCombination":
            coefs = np.reshape(PredictVector[0][9:13], (4, 1))
            # Get the first Patch, NOTE: Use the ShiftedInitConfig, Further: v1...v4 are horizontally stacked, will be transposed in the transformation function
            FirstPatch = shiftedInitConfig["ContactSurfs"][0]
            # Recover to Local Obj
            P_localobj = ConvexCombination_to_Point3D(
                Coef=coefs, ContactSurf=FirstPatch)
            # Convert to Local Obj
            localobj["Px_obj"] = P_localobj[0]
            localobj["Py_obj"] = P_localobj[1]
            localobj["Pz_obj"] = P_localobj[2]
        elif ContactParameterizationType == "FollowRectangelBorder":
            coefs = np.reshape(PredictVector[0][9:11], (2, 1))
            # Get the first patch, NOTE: Use the ShiftedInitConfig, Further: v1...v4 are horizontally stacked, will be transposed in the transformation function
            FirstPatch = shiftedInitConfig["ContactSurfs"][0]
            #   x and y vector for the terrain in world frame (i.e. original world frame, shifted world frame)
            terrain_x_vec = np.array(
                [FirstPatch[0]]).T - np.array([FirstPatch[1]]).T
            terrain_y_vec = np.array(
                [FirstPatch[1]]).T - np.array([FirstPatch[2]]).T
            # Get P_land (Need to add origin and then add the vertial and horizontal displacement)
            P_localobj = np.array([FirstPatch[2]]).T + np.hstack((terrain_x_vec, terrain_y_vec))@coefs
            localobj["Px_obj"] = P_localobj[0]
            localobj["Py_obj"] = P_localobj[1]
            localobj["Pz_obj"] = P_localobj[2]
        else:
            raise Exception(
                "Undefined Contact Location Parameterization Type (ContactParameterizationType)")

        # Make Contacty Timing
        # Timing is always from the back of the index

        InitDS_Dur = PredictVector[0][-3]
        SS_Dur = PredictVector[0][-2]
        DS_Dur = PredictVector[0][-1]

        localobj["InitDS_Ts_obj"] = 0 + InitDS_Dur
        localobj["SS_Ts_obj"] = InitDS_Dur + SS_Dur
        localobj["DS_Ts_obj"] = InitDS_Dur + SS_Dur + DS_Dur

        # localobj["InitDS_Ts_obj"] = PredictVector[0][-3]
        # localobj["SS_Ts_obj"] = PredictVector[0][-2]
        # localobj["DS_Ts_obj"] = PredictVector[0][-1]

    elif Mode == "kNN":  # NOTE: Everything in Local Frame

        print("--------------")
        print("Get Local Obj from kNN")
        print("--------------")

        # Build kNN
        from sklearn.neighbors import KNeighborsRegressor
        K = 5  # number of neighbors
        knn = KNeighborsRegressor(K)

        # Get DataSet
        #DataSetFile = "/home/jiayu/Desktop/MLP_DataSet/Rubbles/" + "DataSet" + "/data"+'.p'

        if DataSetPath == None:
            raise Exception("Unprovided DataSet Path")
        datasetfile = DataSetPath + '/' + 'data.p'
        dataset = pickle.load(open(datasetfile, "rb"))
        x_train = dataset["input"]
        y_train = dataset["output"]

        # Fit kNN
        knn.fit(x_train, y_train)

        # Build Input Vector (Need to Shift Vectors according to shift_world_frame_mode)
        shiftedInitConfig, shiftedTerminalConfig = shiftInitTerminalConfig_to_LocalFrame(
            InitConfig=InitConfig, TerminalConfig=None, Local_Frame_Selection=shift_world_frame)

        if shift_world_frame == None:  # No transformation to local frames, then full input vector
            InputVector = (shiftedInitConfig["x_init"],      shiftedInitConfig["y_init"],      shiftedInitConfig["z_init"],
                           shiftedInitConfig["xdot_init"],   shiftedInitConfig["ydot_init"],   shiftedInitConfig["zdot_init"],
                           shiftedInitConfig["Lx_init"],     shiftedInitConfig["Ly_init"],     shiftedInitConfig["Lz_init"],
                           shiftedInitConfig["PLx_init"],    shiftedInitConfig["PLy_init"],    shiftedInitConfig["PLz_init"],
                           shiftedInitConfig["PRx_init"],    shiftedInitConfig["PRy_init"],    shiftedInitConfig["PRz_init"],
                           shiftedInitConfig["LeftSwingFlag"],
                           shiftedInitConfig["LeftInitSurf"], shiftedInitConfig["RightInitSurf"],
                           shiftedInitConfig["PreviewSurfs_for_Prediction"])
        elif shift_world_frame == "StanceFoot":
            # Decide Contact Location before Swing, NOTE: when in StanceFoot frame, the stationary foot location is always 0, so we ignore that
            # Swing Left Foot
            if (shiftedInitConfig["LeftSwingFlag"] == 1) and (shiftedInitConfig["RightSwingFlag"] == 0):
                contact_before_swing_X = shiftedInitConfig["PLx_init"]
                contact_before_swing_Y = shiftedInitConfig["PLy_init"]
                contact_before_swing_Z = shiftedInitConfig["PLz_init"]
            # Swing Right Foot
            elif (shiftedInitConfig["LeftSwingFlag"] == 0) and (shiftedInitConfig["RightSwingFlag"] == 1):
                contact_before_swing_X = shiftedInitConfig["PRx_init"]
                contact_before_swing_Y = shiftedInitConfig["PRy_init"]
                contact_before_swing_Z = shiftedInitConfig["PRz_init"]
            else:
                raise Exception("Unknow Leg Swing Indicators")
            # Collect Input Vector
            InputVector = (shiftedInitConfig["x_init"],      shiftedInitConfig["y_init"],      shiftedInitConfig["z_init"],
                           shiftedInitConfig["xdot_init"],   shiftedInitConfig["ydot_init"],   shiftedInitConfig["zdot_init"],
                           shiftedInitConfig["Lx_init"],     shiftedInitConfig["Ly_init"],     shiftedInitConfig["Lz_init"],
                           contact_before_swing_X,           contact_before_swing_Y,           contact_before_swing_Z,
                           shiftedInitConfig["LeftSwingFlag"],
                           shiftedInitConfig["LeftInitSurf"], shiftedInitConfig["RightInitSurf"],
                           shiftedInitConfig["PreviewSurfs_for_Prediction"])
        else:
            raise Exception("Unknown Mode of Shifting to Local Frame")

        # Make the input into vector form
        InputVector = np.array([np.concatenate(InputVector, axis=None)])

        # NOTE: Scale the input vector if we need to
        if dataset["PreProcessMode"] == "Standarization" or dataset["PreProcessMode"] == "MaxAbs":
            print("Transfrom Input from Original Form into: ",
                  dataset["PreProcessMode"])
            InputVector = dataset["Scaler_X"].transform(InputVector)
        elif dataset["PreProcessMode"] == "OriginalForm":
            print("Data Represented in Original Form, No need to transform")
        else:
            print("Unknown Pre Processing Model")

        # Predict and Scale Output, if we have
        PredictVector = knn.predict(InputVector)

        # NOTE: Transform back to original form if we need to
        if dataset["PreProcessMode"] == "Standarization" or dataset["PreProcessMode"] == "MaxAbs":
            print("Transfrom Output from: ",
                  dataset["PreProcessMode"], ", back to Original Form")
            PredictVector = dataset["Scaler_Y"].inverse_transform(
                PredictVector)
        elif dataset["PreProcessMode"] == "OriginalForm":
            print("Data Represented in Original Form, No need to transform")
        else:
            print("Unknown Pre Processing Model")

        # Make Local Obj Dict (NOTE: the PredictVector is a 2D array)
        localobj["x_obj"] = PredictVector[0][0]
        localobj["y_obj"] = PredictVector[0][1]
        localobj["z_obj"] = PredictVector[0][2]
        localobj["xdot_obj"] = PredictVector[0][3]
        localobj["ydot_obj"] = PredictVector[0][4]
        localobj["zdot_obj"] = PredictVector[0][5]
        localobj["Lx_obj"] = PredictVector[0][6]
        localobj["Ly_obj"] = PredictVector[0][7]
        localobj["Lz_obj"] = PredictVector[0][8]

        #   Make Contact location
        if ContactParameterizationType == "3DPoints":
            localobj["Px_obj"] = PredictVector[0][9]
            localobj["Py_obj"] = PredictVector[0][10]
            localobj["Pz_obj"] = PredictVector[0][11]
        elif ContactParameterizationType == "ConvexCombination":
            coefs = np.reshape(PredictVector[0][9:13], (4, 1))
            # Get the first Patch, NOTE: Use the ShiftedInitConfig, Further: v1...v4 are horizontally stacked, will be transposed in the transformation function
            FirstPatch = shiftedInitConfig["ContactSurfs"][0]
            # Recover to Local Obj
            P_localobj = ConvexCombination_to_Point3D(
                Coef=coefs, ContactSurf=FirstPatch)
            # Convert to Local Obj
            localobj["Px_obj"] = P_localobj[0]
            localobj["Py_obj"] = P_localobj[1]
            localobj["Pz_obj"] = P_localobj[2]
        elif ContactParameterizationType == "FollowRectangelBorder":
            coefs = np.reshape(PredictVector[0][9:11], (2, 1))
            # Get the first patch, NOTE: Use the ShiftedInitConfig, Further: v1...v4 are horizontally stacked, will be transposed in the transformation function
            FirstPatch = shiftedInitConfig["ContactSurfs"][0]
            #   x and y vector for the terrain in world frame (i.e. original world frame, shifted world frame)
            terrain_x_vec = np.array(
                [FirstPatch[0]]).T - np.array([FirstPatch[1]]).T
            terrain_y_vec = np.array(
                [FirstPatch[1]]).T - np.array([FirstPatch[2]]).T
            # Get P_land (Need to add origin and then add the vertial and horizontal displacement)
            P_localobj = np.array([FirstPatch[2]]).T + np.hstack((terrain_x_vec, terrain_y_vec))@coefs
            localobj["Px_obj"] = P_localobj[0]
            localobj["Py_obj"] = P_localobj[1]
            localobj["Pz_obj"] = P_localobj[2]
        else:
            raise Exception(
                "Undefined Contact Location Parameterization Type (ContactParameterizationType)")

        # Make Contacty Timing
        # Timing is always from the back of the index

        InitDS_Dur = PredictVector[0][-3]
        SS_Dur = PredictVector[0][-2]
        DS_Dur = PredictVector[0][-1]

        localobj["InitDS_Ts_obj"] = 0 + InitDS_Dur
        localobj["SS_Ts_obj"] = InitDS_Dur + SS_Dur
        localobj["DS_Ts_obj"] = InitDS_Dur + SS_Dur + DS_Dur

        # localobj["InitDS_Ts_obj"] = PredictVector[0][-3]
        # localobj["SS_Ts_obj"] = PredictVector[0][-2]
        # localobj["DS_Ts_obj"] = PredictVector[0][-1]

    elif Mode == None:  # Give null/random local obj
        localobj["x_obj"] = 0.0
        localobj["y_obj"] = 0.0
        localobj["z_obj"] = 0.0
        localobj["xdot_obj"] = 0.0
        localobj["ydot_obj"] = 0.0
        localobj["zdot_obj"] = 0.0
        localobj["Lx_obj"] = 0.0
        localobj["Ly_obj"] = 0.0
        localobj["Lz_obj"] = 0.0
        localobj["Px_obj"] = 0.0
        localobj["Py_obj"] = 0.0
        localobj["Pz_obj"] = 0.0
        localobj["InitDS_Ts_obj"] = 0.0
        localobj["SS_Ts_obj"] = 0.0
        localobj["DS_Ts_obj"] = 0.0

    else:
        raise Exception("Unknown Mode for Getting Local Objective")

    return localobj


# Add Noise to Local obj, No noise level on Angular momentum for now
# Noise Level in meters
def add_Noise_to_Localobj(LocalObj=None, InitConfig=None, noiseLevel=0.003):

    # For CoM x pos
    shiftValue = np.random.uniform(-noiseLevel, noiseLevel)
    LocalObj["x_obj"] = LocalObj["x_obj"] + shiftValue

    # For CoM y pos
    shiftValue = np.random.uniform(-noiseLevel, noiseLevel)
    LocalObj["y_obj"] = LocalObj["y_obj"] + shiftValue

    # For CoM z pos
    shiftValue = np.random.uniform(-noiseLevel, noiseLevel)
    LocalObj["z_obj"] = LocalObj["z_obj"] + shiftValue

    # For CoM x velo
    shiftValue = np.random.uniform(-noiseLevel, noiseLevel)
    LocalObj["xdot_obj"] = LocalObj["xdot_obj"] + shiftValue

    # For CoM y Velo
    shiftValue = np.random.uniform(-noiseLevel, noiseLevel)
    LocalObj["ydot_obj"] = LocalObj["ydot_obj"] + shiftValue

    # For CoM z Velo
    shiftValue = np.random.uniform(-noiseLevel, noiseLevel)
    LocalObj["zdot_obj"] = LocalObj["zdot_obj"] + shiftValue

    # NOTE: For angular momentum, we dont add noise for now

    # For Target Contact Location
    #   Noise move along patch length and width axis
    shiftValue_X = np.random.uniform(-noiseLevel, noiseLevel)
    shiftValue_Y = np.random.uniform(-noiseLevel, noiseLevel)
    P_original = np.concatenate(
        [LocalObj["Px_obj"], LocalObj["Py_obj"], LocalObj["Pz_obj"]], axis=None)
    P_shifted = P_original + shiftValue_X * \
        InitConfig["SurfTangentsX"][0] + \
        shiftValue_Y*InitConfig["SurfTangentsY"][0]

    #   Get 4 vertices of the shifted contact location
    P1_shifted = P_shifted + 0.11 * \
        InitConfig["SurfTangentsX"][0] + 0.06*InitConfig["SurfTangentsY"][0]
    P2_shifted = P_shifted + 0.11 * \
        InitConfig["SurfTangentsX"][0] - 0.06*InitConfig["SurfTangentsY"][0]
    P3_shifted = P_shifted - 0.11 * \
        InitConfig["SurfTangentsX"][0] + 0.06*InitConfig["SurfTangentsY"][0]
    P4_shifted = P_shifted - 0.11 * \
        InitConfig["SurfTangentsX"][0] - 0.06*InitConfig["SurfTangentsY"][0]

    #   Get First Contact Patch
    FirstPatch = InitConfig["ContactSurfs"][0]

    # overshoot from right most border  or overshoot from left most border
    if P1_shifted[0] > FirstPatch[0][0] or P3_shifted[0] < FirstPatch[1][0]:
        P_shifted = P_original - shiftValue_X*InitConfig["SurfTangentsX"][0]

    if P1_shifted[1] > FirstPatch[0][1] or P4_shifted[1] < FirstPatch[2][1]:
        P_shifted = P_original - shiftValue_Y*InitConfig["SurfTangentsY"][0]

    LocalObj["Px_obj"] = P_shifted[0]
    LocalObj["Py_obj"] = P_shifted[1]
    LocalObj["Pz_obj"] = P_shifted[2]

    return LocalObj
