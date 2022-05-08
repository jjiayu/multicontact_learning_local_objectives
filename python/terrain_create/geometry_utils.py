import numpy as np
from scipy.spatial.transform import Rotation as R


def getCenter(Surface=None):
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4

    p1 = Surface[0]
    p2 = Surface[1]
    p3 = Surface[2]
    p4 = Surface[3]

    # decide x and y
    center_x = p2[0] + np.abs((p1[0] - p2[0])/2)
    center_y = p4[1] + np.abs((p4[1] - p1[1])/2)

    # decide z
    if p1[2] == p2[2] and p1[2] == p4[2]:  # flat patch
        center_z = p1[2]
    # tilt arond Y axis
    elif p1[2] == p4[2] and p2[2] == p3[2] and (not p1[2]-p2[2] == 0) and (not p4[2]-p3[2] == 0):
        #low + diff/2
        center_z = np.min([p1[2], p2[2]]) + np.abs((p1[2] - p2[2])/2.0)
    # tilt around X axis
    elif p1[2] == p2[2] and p3[2] == p4[2] and (not p2[2]-p3[2] == 0) and (not p1[2]-p4[2] == 0):
        #low + diff/2
        center_z = np.min([p1[2], p4[2]]) + np.abs((p1[2] - p4[2])/2.0)
    #tilt around diagnol X                        or diagonal Y
    elif (p1[2] == p3[2]) and (not p2[2] == p4[2]) or ((p2[2] == p4[2]) and (not p1[2] == p3[2])):
        min_height = np.min([[p1[2], p2[2], p3[2], p4[2]]])
        max_height = np.max([[p1[2], p2[2], p3[2], p4[2]]])
        center_z = min_height + np.abs((max_height - min_height)/2.0)
    else:
        raise Exception("Unknown Patch Type")

    return center_x, center_y, center_z


def getTerrainRotationAngle(Patch):  # Note: Absolute Number only

    # Input Format
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4

    p1 = Patch[0]
    p2 = Patch[1]
    p3 = Patch[2]
    p4 = Patch[3]

    # Patch Length = abs([p1][x] - [p2][x])
    patchLength = np.abs(p1[0] - p2[0])
    # Patch Width = abs([p1][y] - [p4][y])
    patchWidth = np.abs(p1[1] - p4[1])

    # Case 1 all flat
    # print(Patch)
    if p1[2] == p2[2] and p2[2] == p3[2] and p3[2] == p4[2] and p4[2] == p1[2]:
        RotationAngle = 0
        print("flat patch, rotation angle = ", RotationAngle)

    # Case 2, tilt arond Y axis
    elif p1[2] == p4[2] and p2[2] == p3[2] and (not p1[2] == p2[2]) and (not p4[2] == p3[2]):
        tiltAngle = np.arctan2(p2[2]-p1[2], p1[0]-p2[0])
        RotationAngle = tiltAngle/np.pi*180
        print("rotation around y-axis patch, rotation angle = ", RotationAngle)

    # Case 3, tilt around X axis
    elif p1[2] == p2[2] and p3[2] == p4[2] and (not p2[2] == p3[2]) and (not p1[2] == p4[2]):
        tiltAngle = np.arctan2(p1[2]-p4[2], p1[1]-p4[1])
        RotationAngle = tiltAngle/np.pi*180
        print("rotation around x-axis patch, rotation angle = ", RotationAngle)
    #tilt around diagnol X
    elif (p1[2] == p3[2]) and (not p2[2] == p4[2]):
        proj_diagonal_length = np.sqrt(patchLength**2 + patchWidth**2)
        #min_height = np.min([[p1[2], p2[2], p3[2], p4[2]]])
        #max_height = np.max([[p1[2], p2[2], p3[2], p4[2]]])
        #half_height = np.abs(max_height - min_height)/2.0
        tiltAngle = np.arctan2((p2[2]-p4[2]), proj_diagonal_length) 
        RotationAngle = tiltAngle/np.pi*180
        print("rotation around Diagonal X-axis patch, rotation angle = ", RotationAngle)
    #tile around diagnal Y
    elif ((p2[2] == p4[2]) and (not p1[2] == p3[2])):
        proj_diagonal_length = np.sqrt(patchLength**2 + patchWidth**2) 
        #min_height = np.min([[p1[2], p2[2], p3[2], p4[2]]])
        #max_height = np.max([[p1[2], p2[2], p3[2], p4[2]]])
        #half_height = np.abs(max_height - min_height)/2.0
        tiltAngle = np.arctan2(p3[2]-p1[2], proj_diagonal_length)
        RotationAngle = tiltAngle/np.pi*180
        print("rotation around Diagonal Y-axis patch, rotation angle = ", RotationAngle)
    else:
        raise Exception("Un-defined Terrain Type")

    return RotationAngle


def getTerrainTagentsNormOrientation(Patch):
    # Input Format
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4

    p1 = Patch[0]
    p2 = Patch[1]
    p3 = Patch[2]
    p4 = Patch[3]

    # Patch Length = abs([p1][x] - [p2][x])
    patchLength = np.abs(p1[0] - p2[0])
    # Patch Width = abs([p1][y] - [p4][y])
    patchWidth = np.abs(p1[1] - p4[1])

    # Unrotated Terrain Norm and Tangents
    TerrainTangentX = np.array([1, 0, 0])
    TerrainTangentY = np.array([0, 1, 0])
    TerrainNorm = np.array([0, 0, 1])
    # Make Flat Orientation (Default)
    r = R.from_euler('y', 0, degrees=False)  # rotate on any axis with 0 degree
    TerrainOrientation = r.as_matrix()

    # Case 1 all flat
    # print(Patch)
    if p1[2] == p2[2] and p2[2] == p3[2] and p3[2] == p4[2] and p4[2] == p1[2]:
        print("ComputeTangentNorm: a Flat Terrain, use the default set up of terrain tangent and norm")
        print("TerrainTangent X: ", TerrainTangentX)
        print("TerrainTangent Y: ", TerrainTangentY)
        print("Terrain Norm: ", TerrainNorm)
        print("Terrain Orientation in Matrix Form: \n", TerrainOrientation)
    # Case 2, tilt arond Y axis
    elif p1[2] == p4[2] and p2[2] == p3[2] and (not p1[2] == p2[2]) and (not p4[2] == p3[2]):
        tiltAngle = np.arctan2(p2[2]-p1[2], p1[0]-p2[0])
        print("ComputeTangentNorm: tilt arond Y axis, tilt angle = ",
              str(tiltAngle/np.pi*180))
        r = R.from_euler('y', tiltAngle, degrees=False)
        TerrainTangentX = r.as_matrix()@TerrainTangentX
        TerrainNorm = r.as_matrix()@TerrainNorm
        TerrainOrientation = r.as_matrix()  # In rotation matrix form
        print("TerrainTangent X: ", TerrainTangentX)
        print("TerrainTangent Y: ", TerrainTangentY)
        print("Terrain Norm: ", TerrainNorm)
        print("ComputeOrientation: Terrain tilt around Y axis, second column should be 0 1 0; \n", TerrainOrientation)
        print("Rechecking: Terrain Tangent *X* angle with respect to the original one: ",
              str(angle_between(np.array([1, 0, 0]), TerrainTangentX)/np.pi*180), "degrees")
        print("Rechecking: Terrain Norm angle with respect to the original one: ", str(
            angle_between(np.array([0, 0, 1]), TerrainNorm)/np.pi*180), "degrees")
        print(" ")

    # Case 3, tilt around X axis
    elif p1[2] == p2[2] and p3[2] == p4[2] and (not p2[2] == p3[2]) and (not p1[2] == p4[2]):
        tiltAngle = np.arctan2(p1[2]-p4[2], p1[1]-p4[1])
        print("ComputeTangentNorm: tilt arond X axis, tilt angle = ",
              str(tiltAngle/np.pi*180))
        r = R.from_euler('x', tiltAngle, degrees=False)
        TerrainTangentY = r.as_matrix()@TerrainTangentY
        TerrainNorm = r.as_matrix()@TerrainNorm
        TerrainOrientation = r.as_matrix()  # In rotation matrix form
        print("TerrainTangent X: ", TerrainTangentX)
        print("TerrainTangent Y: ", TerrainTangentY)
        print("Terrain Norm: ", TerrainNorm)
        print("ComputeOrientation: Terrain tilt around X axis, first column should be 1 0 0; \n", TerrainOrientation)
        print("Rechecking: Terrain Tangent *Y* angle with respect to the original one: ",
              str(angle_between(np.array([0, 1, 0]), TerrainTangentY)/np.pi*180), "degrees")
        print("Rechecking: Terrain Norm angle with respect to the original one: ", str(
            angle_between(np.array([0, 0, 1]), TerrainNorm)/np.pi*180), "degrees")
        print(" ")

    #tilt around diagnol X                        
    elif (p1[2] == p3[2]) and (not p2[2] == p4[2]):
        #rotate terrain norm first
        proj_diagonal_length = np.sqrt(patchLength**2 + patchWidth**2)
        #min_height = np.min([[p1[2], p2[2], p3[2], p4[2]]])
        #max_height = np.max([[p1[2], p2[2], p3[2], p4[2]]])
        #half_height = np.abs(max_height - min_height)/2.0
        tiltAngle = np.arctan2((p2[2]-p4[2]), proj_diagonal_length)
        print("ComputeTangentNorm: tilt arond Diagonal X axis, tilt angle = ",
              str(tiltAngle/np.pi*180))
        print("!!!: ", np.arctan2(p2[2]-p1[2],0.4)/np.pi*180)
        print("!!!: ", np.arctan2(p2[2]-p3[2],0.4)/np.pi*180)
        print("!!!: ", np.arctan2(p3[2]-p4[2],0.4)/np.pi*180)
        print("!!!: ", np.arctan2(p1[2]-p4[2],0.4)/np.pi*180)
        yaw_tilt_angle = np.arctan2(patchWidth,patchLength)
        r = R.from_euler('zxy', [yaw_tilt_angle, tiltAngle, 0.0], degrees=False)
        TerrainNorm = r.as_matrix()@TerrainNorm
        
        #Get terrain tangent X
        rot_angle_for_x = np.arctan2(p3[2]-p4[2],patchLength)
        rotation_for_x = R.from_euler('y', rot_angle_for_x , degrees=False)
        TerrainTangentX = rotation_for_x.as_matrix()@TerrainTangentX

        #get terrain tangent Y
        rot_angle_for_y = np.arctan2(p2[2]-p3[2],patchWidth)
        rotation_for_y = R.from_euler('x', rot_angle_for_y , degrees=False)
        TerrainTangentY = rotation_for_y.as_matrix()@TerrainTangentY

        #TerrainTangentX = r.as_matrix()@TerrainTangentX
        #TerrainTangentY = r.as_matrix()@TerrainTangentY
        
        TerrainOrientation = np.array([TerrainTangentX,TerrainTangentY,TerrainNorm]).T
        #r.as_matrix()  # In rotation matrix form

        print("TerrainTangent X: ", TerrainTangentX)
        print("TerrainTangent Y: ", TerrainTangentY)
        print("Terrain Norm: ", TerrainNorm)
        print("Terrain Orientation: ", TerrainOrientation)
    #tile around diagnal Y
    elif ((p2[2] == p4[2]) and (not p1[2] == p3[2])):
        #rotate terrain norm first
        proj_diagonal_length = np.sqrt(patchLength**2 + patchWidth**2)
        #min_height = np.min([[p1[2], p2[2], p3[2], p4[2]]])
        #max_height = np.max([[p1[2], p2[2], p3[2], p4[2]]])
        #half_height = np.abs(max_height - min_height)/2.0
        tiltAngle = np.arctan2((p3[2]-p1[2]), proj_diagonal_length)
        print("ComputeTangentNorm: tilt arond Diagonal Y axis, tilt angle = ",
              str(tiltAngle/np.pi*180))
        print("!!!: ", np.arctan2(p2[2]-p1[2],0.4)/np.pi*180)
        print("!!!: ", np.arctan2(p2[2]-p3[2],0.4)/np.pi*180)
        print("!!!: ", np.arctan2(p3[2]-p4[2],0.4)/np.pi*180)
        print("!!!: ", np.arctan2(p1[2]-p4[2],0.4)/np.pi*180)
        yaw_tilt_angle = np.arctan2(patchLength,patchWidth)
        r = R.from_euler('zyx', [yaw_tilt_angle, tiltAngle, 0.0], degrees=False)
        TerrainNorm = r.as_matrix()@TerrainNorm
        
        #Get terrain tangent X
        rot_angle_for_x = np.arctan2(p3[2]-p4[2],patchLength)
        rotation_for_x = R.from_euler('y', rot_angle_for_x , degrees=False)
        TerrainTangentX = rotation_for_x.as_matrix()@TerrainTangentX

        #get terrain tangent Y
        rot_angle_for_y = np.arctan2(p2[2]-p3[2],patchWidth)
        rotation_for_y = R.from_euler('x', rot_angle_for_y , degrees=False)
        TerrainTangentY = rotation_for_y.as_matrix()@TerrainTangentY

        #TerrainTangentX = r.as_matrix()@TerrainTangentX
        #TerrainTangentY = r.as_matrix()@TerrainTangentY
        
        TerrainOrientation = np.array([TerrainTangentX,TerrainTangentY,TerrainNorm]).T
        #r.as_matrix()  # In rotation matrix form

        print("TerrainTangent X: ", TerrainTangentX)
        print("TerrainTangent Y: ", TerrainTangentY)
        print("Terrain Norm: ", TerrainNorm)
        print("Terrain Orientation: ", TerrainOrientation)

        
    else:
        raise Exception("Un-defined Terrain Type")

    return TerrainTangentX, TerrainTangentY, TerrainNorm, TerrainOrientation


def getSurfaceType(Patch):
    # Input Format
    # p2---------------------p1
    # |                      |
    # |                      |
    # |                      |
    # p3---------------------p4
    # Get Vertices
    p1 = Patch[0]
    p2 = Patch[1]
    p3 = Patch[2]
    p4 = Patch[3]

    # Patch Length = abs([p1][x] - [p2][x])
    patchLength = np.abs(p1[0] - p2[0])
    # Patch Width = abs([p1][y] - [p4][y])
    patchWidth = np.abs(p1[1] - p4[1])

    if p1[2] == p2[2] and p2[2] == p3[2] and p3[2] == p4[2] and p4[2] == p1[2]:  # Flat Patch
        terrainType = "flat"
    # Case 2, tilt arond Y axis
    elif p1[2] == p4[2] and p2[2] == p3[2] and (not p1[2] == p2[2]) and (not p4[2] == p3[2]):
        tiltAngle = np.arctan2(p2[2]-p1[2], p1[0]-p2[0])
        if tiltAngle < 0:
            terrainType = "Y_negative"
        elif tiltAngle > 0:
            terrainType = "Y_positive"
        else:
            raise Exception("Cannot Identify Surface Type")
    # Case 3, tilt around X axis
    elif p1[2] == p2[2] and p3[2] == p4[2] and (not p2[2] == p3[2]) and (not p1[2] == p4[2]):
        tiltAngle = np.arctan2(p1[2]-p4[2], p1[1]-p4[1])
        if tiltAngle < 0:
            terrainType = "X_negative"
        elif tiltAngle > 0:
            terrainType = "X_positive"
        else:
            raise Exception("Cannot Identify Surface Type")
    #tilt around diagnol X                        
    elif (p1[2] == p3[2]) and (not p2[2] == p4[2]):
        proj_diagonal_length = np.sqrt(patchLength**2 + patchWidth**2)
        #min_height = np.min([[p1[2], p2[2], p3[2], p4[2]]])
        #max_height = np.max([[p1[2], p2[2], p3[2], p4[2]]])
        #half_height = np.abs(max_height - min_height)/2.0
        tiltAngle = np.arcsin(p2[2]-p4[2]/proj_diagonal_length)
        if tiltAngle < 0:
            terrainType = "DiagX_negative"
        elif tiltAngle > 0:
            terrainType = "DiagX_positive"
        else:
            raise Exception("Cannot Identify Surface Type")
    #tilt around diagnal Y
    elif ((p2[2] == p4[2]) and (not p1[2] == p3[2])):
        proj_diagonal_length = np.sqrt(patchLength**2 + patchWidth**2)
        #min_height = np.min([[p1[2], p2[2], p3[2], p4[2]]])
        #max_height = np.max([[p1[2], p2[2], p3[2], p4[2]]])
        #half_height = np.abs(max_height - min_height)/2.0
        tiltAngle = np.arcsin(p3[2]-p1[2]/proj_diagonal_length)
        if tiltAngle < 0:
            terrainType = "DiagY_negative"
        elif tiltAngle > 0:
            terrainType = "DiagY_positive"
        else:
            raise Exception("Cannot Identify Surface Type")
    else:
        raise Exception("Unknow Terrain Type")

    return terrainType


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
