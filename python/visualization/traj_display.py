import numpy as np
import pickle

import matplotlib.pyplot as plt #Matplotlib
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4)


def draw_timeSeries_and_traj(filePath = None, query_traj = None, traj_color = None, startStepNum = 0, EndStepNum = 14, NumLocalKonts = 7, fig= None, ax = None):

    #Load the File
    with open(filePath, 'rb') as f:
        data = pickle.load(f)

    #Get index
    Level1_VarIndex = data["VarIdx_of_All_Levels"]["Level1_Var_Index"]
    Level2_VarIndex = data["VarIdx_of_All_Levels"]["Level2_Var_Index"]

    #Get Optimization Result
    OptResults = data["SingleOptResultSavings"]
    CasadiParameters = data["CasadiParameters"]

    #Build result containers

    TISD_Trajectories = []

    x_result = [];       y_result = [];       z_result = []
    xdot_result = [];    ydot_result = [];    zdot_result = []
    Lx_result = [];      Ly_result = [];      Lz_result = []
    Ldotx_result = [];   Ldoty_result = [];   Ldotz_result = []

    FL1x_res = [];       FL2x_res = [];       FL3x_res = [];      FL4x_res= []
    FL1y_res = [];       FL2y_res = [];       FL3y_res = [];      FL4y_res= []
    FL1z_res = [];       FL2z_res = [];       FL3z_res = [];      FL4z_res= []

    FR1x_res = [];       FR2x_res = [];       FR3x_res = [];      FR4x_res = []
    FR1y_res = [];       FR2y_res = [];       FR3y_res = [];      FR4y_res = []
    FR1z_res = [];       FR2z_res = [];       FR3z_res = [];      FR4z_res = []

    px_List = [];     py_List = [];     pz_List = []

    FLx_res = [];     FLy_res = [];     FLz_res = []
    FRx_res = [];     FRy_res = [];     FRz_res = []

    Fx_res = [];      Fy_res = [];      Fz_res = []

    Ts_List = []
    timeseries = []
    time_offset = 0

    for roundIdx in range(len(OptResults)):

        traj = OptResults[roundIdx]["opt_res"]
        casadiParams = CasadiParameters[roundIdx]

        #Get raw data
        x_traj = traj[Level1_VarIndex["x"][0]:Level1_VarIndex["x"][1]+1]
        y_traj = traj[Level1_VarIndex["y"][0]:Level1_VarIndex["y"][1]+1]
        z_traj = traj[Level1_VarIndex["z"][0]:Level1_VarIndex["z"][1]+1]
        xdot_traj = traj[Level1_VarIndex["xdot"][0]:Level1_VarIndex["xdot"][1]+1]
        ydot_traj = traj[Level1_VarIndex["ydot"][0]:Level1_VarIndex["ydot"][1]+1]
        zdot_traj = traj[Level1_VarIndex["zdot"][0]:Level1_VarIndex["zdot"][1]+1]

        Lx_traj = traj[Level1_VarIndex["Lx"][0]:Level1_VarIndex["Lx"][1]+1]
        Ly_traj = traj[Level1_VarIndex["Ly"][0]:Level1_VarIndex["Ly"][1]+1]
        Lz_traj = traj[Level1_VarIndex["Lz"][0]:Level1_VarIndex["Lz"][1]+1]
        Ldotx_traj = traj[Level1_VarIndex["Ldotx"][0]:Level1_VarIndex["Ldotx"][1]+1]
        Ldoty_traj = traj[Level1_VarIndex["Ldoty"][0]:Level1_VarIndex["Ldoty"][1]+1]
        Ldotz_traj = traj[Level1_VarIndex["Ldotz"][0]:Level1_VarIndex["Ldotz"][1]+1]

        FL1x_traj = traj[Level1_VarIndex["FL1x"][0]:Level1_VarIndex["FL1x"][1]+1]
        FL2x_traj = traj[Level1_VarIndex["FL2x"][0]:Level1_VarIndex["FL2x"][1]+1]
        FL3x_traj = traj[Level1_VarIndex["FL3x"][0]:Level1_VarIndex["FL3x"][1]+1]
        FL4x_traj = traj[Level1_VarIndex["FL4x"][0]:Level1_VarIndex["FL4x"][1]+1]

        FR1x_traj = traj[Level1_VarIndex["FR1x"][0]:Level1_VarIndex["FR1x"][1]+1]
        FR2x_traj = traj[Level1_VarIndex["FR2x"][0]:Level1_VarIndex["FR2x"][1]+1]
        FR3x_traj = traj[Level1_VarIndex["FR3x"][0]:Level1_VarIndex["FR3x"][1]+1]
        FR4x_traj = traj[Level1_VarIndex["FR4x"][0]:Level1_VarIndex["FR4x"][1]+1]

        FL1y_traj = traj[Level1_VarIndex["FL1y"][0]:Level1_VarIndex["FL1y"][1]+1]
        FL2y_traj = traj[Level1_VarIndex["FL2y"][0]:Level1_VarIndex["FL2y"][1]+1]
        FL3y_traj = traj[Level1_VarIndex["FL3y"][0]:Level1_VarIndex["FL3y"][1]+1]
        FL4y_traj = traj[Level1_VarIndex["FL4y"][0]:Level1_VarIndex["FL4y"][1]+1]

        FR1y_traj = traj[Level1_VarIndex["FR1y"][0]:Level1_VarIndex["FR1y"][1]+1]
        FR2y_traj = traj[Level1_VarIndex["FR2y"][0]:Level1_VarIndex["FR2y"][1]+1]
        FR3y_traj = traj[Level1_VarIndex["FR3y"][0]:Level1_VarIndex["FR3y"][1]+1]
        FR4y_traj = traj[Level1_VarIndex["FR4y"][0]:Level1_VarIndex["FR4y"][1]+1]

        FL1z_traj = traj[Level1_VarIndex["FL1z"][0]:Level1_VarIndex["FL1z"][1]+1]
        FL2z_traj = traj[Level1_VarIndex["FL2z"][0]:Level1_VarIndex["FL2z"][1]+1]
        FL3z_traj = traj[Level1_VarIndex["FL3z"][0]:Level1_VarIndex["FL3z"][1]+1]
        FL4z_traj = traj[Level1_VarIndex["FL4z"][0]:Level1_VarIndex["FL4z"][1]+1]

        FR1z_traj = traj[Level1_VarIndex["FR1z"][0]:Level1_VarIndex["FR1z"][1]+1]
        FR2z_traj = traj[Level1_VarIndex["FR2z"][0]:Level1_VarIndex["FR2z"][1]+1]
        FR3z_traj = traj[Level1_VarIndex["FR3z"][0]:Level1_VarIndex["FR3z"][1]+1]
        FR4z_traj = traj[Level1_VarIndex["FR4z"][0]:Level1_VarIndex["FR4z"][1]+1]

        FLx_traj = FL1x_traj + FL2x_traj + FL3x_traj + FL4x_traj
        FLy_traj = FL1y_traj + FL2y_traj + FL3y_traj + FL4y_traj
        FLz_traj = FL1z_traj + FL2z_traj + FL3z_traj + FL4z_traj

        FRx_traj = FR1x_traj + FR2x_traj + FR3x_traj + FR4x_traj
        FRy_traj = FR1y_traj + FR2y_traj + FR3y_traj + FR4y_traj
        FRz_traj = FR1z_traj + FR2z_traj + FR3z_traj + FR4z_traj

        Fx_traj = FLx_traj + FRx_traj
        Fy_traj = FLy_traj + FRy_traj
        Fz_traj = FLz_traj + FRz_traj

        px_res = traj[Level1_VarIndex["px"][0]:Level1_VarIndex["px"][1]+1]
        py_res = traj[Level1_VarIndex["py"][0]:Level1_VarIndex["py"][1]+1]
        pz_res = traj[Level1_VarIndex["pz"][0]:Level1_VarIndex["pz"][1]+1]

        Ts_res = traj[Level1_VarIndex["Ts"][0]:Level1_VarIndex["Ts"][1]+1]
        #Ts_level2_res = traj[Level2_VarIndex["Ts"][0]:Level2_VarIndex["Ts"][1]+1]

        #Initial Conditions
        LeftSwingFlag = casadiParams[0]
        RightSwingFlag = casadiParams[1]

        PLx_init = casadiParams[14]
        PLy_init = casadiParams[15]
        PLz_init = casadiParams[16]

        PRx_init = casadiParams[17]
        PRy_init = casadiParams[18]
        PRz_init = casadiParams[19]

        #get traj for each phase
        Phase1_TimeSeries = np.linspace(0,Ts_res[0],NumLocalKonts+1)
        Phase2_TimeSeries = np.linspace(Ts_res[0],Ts_res[1], NumLocalKonts+1)
        Phase3_TimeSeries = np.linspace(Ts_res[1],Ts_res[2], NumLocalKonts+1)
        timeline = np.concatenate((time_offset+Phase1_TimeSeries,time_offset+Phase2_TimeSeries[1:],time_offset+Phase3_TimeSeries[1:]),axis=None)
        time_offset = time_offset+Phase3_TimeSeries[-1]

        Phase1_x = x_traj[0:NumLocalKonts+1];   Phase2_x = x_traj[NumLocalKonts:2*NumLocalKonts+1];   Phase3_x = x_traj[2*NumLocalKonts:]
        Phase1_y = y_traj[0:NumLocalKonts+1];   Phase2_y = y_traj[NumLocalKonts:2*NumLocalKonts+1];   Phase3_y = y_traj[2*NumLocalKonts:]
        Phase1_z = z_traj[0:NumLocalKonts+1];   Phase2_z = z_traj[NumLocalKonts:2*NumLocalKonts+1];   Phase3_z = z_traj[2*NumLocalKonts:]

        Phase1_xdot = xdot_traj[0:NumLocalKonts+1];   Phase2_xdot = xdot_traj[NumLocalKonts:2*NumLocalKonts+1];   Phase3_xdot = xdot_traj[2*NumLocalKonts:]
        Phase1_ydot = ydot_traj[0:NumLocalKonts+1];   Phase2_ydot = ydot_traj[NumLocalKonts:2*NumLocalKonts+1];   Phase3_ydot = ydot_traj[2*NumLocalKonts:]
        Phase1_zdot = zdot_traj[0:NumLocalKonts+1];   Phase2_zdot = zdot_traj[NumLocalKonts:2*NumLocalKonts+1];   Phase3_zdot = zdot_traj[2*NumLocalKonts:]

        Phase1_Lx = Lx_traj[0:NumLocalKonts+1];   Phase2_Lx = Lx_traj[NumLocalKonts:2*NumLocalKonts+1];   Phase3_Lx = Lx_traj[2*NumLocalKonts:]
        Phase1_Ly = Ly_traj[0:NumLocalKonts+1];   Phase2_Ly = Ly_traj[NumLocalKonts:2*NumLocalKonts+1];   Phase3_Ly = Ly_traj[2*NumLocalKonts:]
        Phase1_Lz = Lz_traj[0:NumLocalKonts+1];   Phase2_Lz = Lz_traj[NumLocalKonts:2*NumLocalKonts+1];   Phase3_Lz = Lz_traj[2*NumLocalKonts:]

        Phase1_Ldotx = Ldotx_traj[0:NumLocalKonts+1];   Phase2_Ldotx = Ldotx_traj[NumLocalKonts:2*NumLocalKonts+1];   Phase3_Ldotx = Ldotx_traj[2*NumLocalKonts:]
        Phase1_Ldoty = Ldoty_traj[0:NumLocalKonts+1];   Phase2_Ldoty = Ldoty_traj[NumLocalKonts:2*NumLocalKonts+1];   Phase3_Ldoty = Ldoty_traj[2*NumLocalKonts:]
        Phase1_Ldotz = Ldotz_traj[0:NumLocalKonts+1];   Phase2_Ldotz = Ldotz_traj[NumLocalKonts:2*NumLocalKonts+1];   Phase3_Ldotz = Ldotz_traj[2*NumLocalKonts:]

        TSIDTrajectory = {}
        
        #Init Double Phase
        TSIDTrajectory["InitDouble_TimeSeries"]=Phase1_TimeSeries
        TSIDTrajectory["InitDouble_x"]=Phase1_x;    TSIDTrajectory["InitDouble_y"]=Phase1_y;   TSIDTrajectory["InitDouble_z"]=Phase1_z
        TSIDTrajectory["InitDouble_Lx"]=Phase1_Lx;  TSIDTrajectory["InitDouble_Ly"]=Phase1_Ly; TSIDTrajectory["InitDouble_Lz"]=Phase1_Lz
        TSIDTrajectory["InitDouble_xdot"]=Phase1_xdot;   TSIDTrajectory["InitDouble_ydot"]=Phase1_ydot;    TSIDTrajectory["InitDouble_zdot"]=Phase1_zdot
        TSIDTrajectory["InitDouble_Ldotx"]=Phase1_Ldotx; TSIDTrajectory["InitDouble_Ldoty"]=Phase1_Ldoty;  TSIDTrajectory["InitDouble_Ldotz"]=Phase1_Ldotz

        #Swing Phase
        TSIDTrajectory["Swing_TimeSeries"]=Phase2_TimeSeries
        TSIDTrajectory["Swing_x"]=Phase2_x;           TSIDTrajectory["Swing_y"]=Phase2_y;          TSIDTrajectory["Swing_z"]=Phase2_z
        TSIDTrajectory["Swing_Lx"]=Phase2_Lx;         TSIDTrajectory["Swing_Ly"]=Phase2_Ly;        TSIDTrajectory["Swing_Lz"]=Phase2_Lz
        TSIDTrajectory["Swing_xdot"]=Phase2_xdot;     TSIDTrajectory["Swing_ydot"]=Phase2_ydot;    TSIDTrajectory["Swing_zdot"]=Phase2_zdot
        TSIDTrajectory["Swing_Ldotx"]=Phase2_Ldotx;   TSIDTrajectory["Swing_Ldoty"]=Phase2_Ldoty;  TSIDTrajectory["Swing_Ldotz"]=Phase2_Ldotz

        #DoubleSupport Phase
        TSIDTrajectory["DoubleSupport_TimeSeries"]=Phase3_TimeSeries
        TSIDTrajectory["DoubleSupport_x"]=Phase3_x;    TSIDTrajectory["DoubleSupport_y"]=Phase3_y;   TSIDTrajectory["DoubleSupport_z"]=Phase3_z
        TSIDTrajectory["DoubleSupport_Lx"]=Phase3_Lx;  TSIDTrajectory["DoubleSupport_Ly"]=Phase3_Ly; TSIDTrajectory["DoubleSupport_Lz"]=Phase3_Lz
        TSIDTrajectory["DoubleSupport_xdot"]=Phase3_xdot;   TSIDTrajectory["DoubleSupport_ydot"]=Phase3_ydot;    TSIDTrajectory["DoubleSupport_zdot"]=Phase3_zdot
        TSIDTrajectory["DoubleSupport_Ldotx"]=Phase3_Ldotx; TSIDTrajectory["DoubleSupport_Ldoty"]=Phase3_Ldoty;  TSIDTrajectory["DoubleSupport_Ldotz"]=Phase3_Ldotz

        #Contact config
        TSIDTrajectory["Init_PL"]=[PLx_init,PLy_init,PLz_init];   TSIDTrajectory["Init_PR"]=[PRx_init,PRy_init,PRz_init]
        TSIDTrajectory["Landing_P"] = list(np.concatenate((px_res,py_res,pz_res),axis=None))
        TSIDTrajectory["LeftSwingFlag"]=LeftSwingFlag
        TSIDTrajectory["RightSwingFlag"]=RightSwingFlag

        TISD_Trajectories.append(TSIDTrajectory)

        #print(Ts_level2_res)
        #print("y motion:", np.max(y_traj) - np.min(y_traj))

        if roundIdx == 0:
            x_result.append(x_traj);          y_result.append(y_traj);           z_result.append(z_traj)
            xdot_result.append(xdot_traj);    ydot_result.append(ydot_traj);     zdot_result.append(zdot_traj)
            Lx_result.append(Lx_traj);        Ly_result.append(Ly_traj);         Lz_result.append(Lz_traj)
            Ldotx_result.append(Ldotx_traj);  Ldoty_result.append(Ldoty_traj);    Ldotz_result.append(Ldotz_traj)
            px_List.append(px_res);           py_List.append(py_res);            pz_List.append(pz_res)
            Ts_List.append(Ts_res)

            FL1x_res.append(FL1x_traj);       FL1y_res.append(FL1y_traj);        FL1z_res.append(FL1z_traj)
            FL2x_res.append(FL2x_traj);       FL2y_res.append(FL2y_traj);        FL2z_res.append(FL2z_traj)
            FL3x_res.append(FL3x_traj);       FL3y_res.append(FL3y_traj);        FL3z_res.append(FL3z_traj)
            FL4x_res.append(FL4x_traj);       FL4y_res.append(FL4y_traj);        FL4z_res.append(FL4z_traj)

            FR1x_res.append(FR1x_traj);       FR1y_res.append(FR1y_traj);        FR1z_res.append(FR1z_traj)
            FR2x_res.append(FR2x_traj);       FR2y_res.append(FR2y_traj);        FR2z_res.append(FR2z_traj)
            FR3x_res.append(FR3x_traj);       FR3y_res.append(FR3y_traj);        FR3z_res.append(FR3z_traj)
            FR4x_res.append(FR4x_traj);       FR4y_res.append(FR4y_traj);        FR4z_res.append(FR4z_traj)

            FLx_res.append(FLx_traj);         FLy_res.append(FLy_traj);          FLz_res.append(FLz_traj)
            FRx_res.append(FRx_traj);         FRy_res.append(FRy_traj);          FRz_res.append(FRz_traj)

            Fx_res.append(Fx_traj);           Fy_res.append(Fy_traj);            Fz_res.append(Fz_traj)

            timeseries.append(timeline)

        else:
            x_result.append(x_traj);          y_result.append(y_traj);           z_result.append(z_traj)
            xdot_result.append(xdot_traj);    ydot_result.append(ydot_traj);     zdot_result.append(zdot_traj)
            Lx_result.append(Lx_traj);        Ly_result.append(Ly_traj);         Lz_result.append(Lz_traj)
            Ldotx_result.append(Ldotx_traj);  Ldoty_result.append(Ldoty_traj);   Ldotz_result.append(Ldotz_traj)
            
            px_List.append(px_res);           py_List.append(py_res);            pz_List.append(pz_res)
            
            Ts_List.append(Ts_res);           timeseries.append(timeline)

            FL1x_res.append(FL1x_traj);       FL1y_res.append(FL1y_traj);        FL1z_res.append(FL1z_traj)
            FL2x_res.append(FL2x_traj);       FL2y_res.append(FL2y_traj);        FL2z_res.append(FL2z_traj)
            FL3x_res.append(FL3x_traj);       FL3y_res.append(FL3y_traj);        FL3z_res.append(FL3z_traj)
            FL4x_res.append(FL4x_traj);       FL4y_res.append(FL4y_traj);        FL4z_res.append(FL4z_traj)

            FR1x_res.append(FR1x_traj);       FR1y_res.append(FR1y_traj);        FR1z_res.append(FR1z_traj)
            FR2x_res.append(FR2x_traj);       FR2y_res.append(FR2y_traj);        FR2z_res.append(FR2z_traj)
            FR3x_res.append(FR3x_traj);       FR3y_res.append(FR3y_traj);        FR3z_res.append(FR3z_traj)
            FR4x_res.append(FR4x_traj);       FR4y_res.append(FR4y_traj);        FR4z_res.append(FR4z_traj)

            FLx_res.append(FLx_traj);         FLy_res.append(FLy_traj);          FLz_res.append(FLz_traj)

            FRx_res.append(FRx_traj);         FRy_res.append(FRy_traj);          FRz_res.append(FRz_traj)
            
            Fx_res.append(Fx_traj);           Fy_res.append(Fy_traj);            Fz_res.append(Fz_traj)

    #Convert to a single numpy array
    px_List = np.concatenate((px_List),axis=None)
    py_List = np.concatenate((py_List),axis=None)
    pz_List = np.concatenate((pz_List),axis=None)

    Ts_List = np.concatenate((Ts_List),axis=None)

    #CoM result
    Traj_Collected = {"timeseries":timeseries,
                      "x_result":x_result,         "y_result":y_result,         "z_result":z_result,
                      "xdot_result":xdot_result,   "ydot_result":ydot_result,   "zdot_result":zdot_result,
                      "Lx_result":Lx_result,       "Ly_result":Ly_result,       "Lz_result":Lz_result,
                      "Ldotx_result":Ldotx_result, "Ldoty_result":Ldoty_result, "Ldotz_result":Ldotz_result,
                      "px_result":px_List,         "py_result":py_List,         "pz_result":pz_List,
                      "Ts_result":Ts_List,
                      "FL1x_result":FL1x_res,      "FL1y_result":FL1y_res,      "FL1z_result":FL1z_res,
                      "FL2x_result":FL2x_res,      "FL2y_result":FL2y_res,      "FL2z_result":FL2z_res,
                      "FL3x_result":FL3x_res,      "FL3y_result":FL3y_res,      "FL3z_result":FL3z_res,
                      "FL4x_result":FL4x_res,      "FL4y_result":FL4y_res,      "FL4z_result":FL4z_res,
                      "FR1x_result":FR1x_res,      "FR1y_result":FR1y_res,      "FR1z_result":FR1z_res,
                      "FR2x_result":FR2x_res,      "FR2y_result":FR2y_res,      "FR2z_result":FR2z_res,
                      "FR3x_result":FR3x_res,      "FR3y_result":FR3y_res,      "FR3z_result":FR3z_res,
                      "FR4x_result":FR4x_res,      "FR4y_result":FR4y_res,      "FR4z_result":FR4z_res,
                      "FLx_result":FLx_res,        "FLy_result":FLy_res,        "FLz_result":FLz_res,
                      "FRx_result":FRx_res,        "FRy_result":FRy_res,        "FRz_result":FRz_res,
                      "Fx_result":Fx_res,          "Fy_result":Fy_res,          "Fz_result":Fz_res,
    }

    for traj_idx in range(startStepNum,EndStepNum):
        if traj_idx%2 == 0:
            if traj_idx == 0:
                if traj_idx <= len(Traj_Collected["x_result"])-1:
                    ax.plot(Traj_Collected["timeseries"][traj_idx],Traj_Collected[query_traj][traj_idx],color = traj_color, label=filePath[-19:],linestyle='dashed')
            else:
                if traj_idx <= len(Traj_Collected["x_result"])-1:
                    ax.plot(Traj_Collected["timeseries"][traj_idx],Traj_Collected[query_traj][traj_idx],color = traj_color,linestyle='dashed')
        else:
            if traj_idx <= len(Traj_Collected["x_result"])-1:
                ax.plot(Traj_Collected["timeseries"][traj_idx],Traj_Collected[query_traj][traj_idx],color = traj_color)


    return None