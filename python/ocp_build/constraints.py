# Module for creating constraints

import numpy as np
import casadi as ca

# Standard Equality Constraint, in the form of a - b = 0, mainly for setting intial and terminal conditions

def std_eq_constraint(SwingLegIndicator=None, a=None, b=None, g=None, glb=None, gub=None):
    if SwingLegIndicator == None:
        g.append(a-b)
        glb.append(np.zeros(a.shape[0]*a.shape[1]))
        gub.append(np.zeros(a.shape[0]*a.shape[1]))
    else:
        g.append(ca.if_else(SwingLegIndicator, a-b,
                 np.zeros(a.shape[0]*a.shape[1])))
        glb.append(np.zeros(a.shape[0]*a.shape[1]))
        gub.append(np.zeros(a.shape[0]*a.shape[1]))
    return g, glb, gub


def slackConstrained_SingleVar(a=None, b=None, slackratio=0.0, g=None, glb=None, gub=None):
    g.append(a-(1.0-slackratio)*b)
    glb.append([0])
    gub.append([np.inf])

    g.append(a-(1.0+slackratio)*b)
    glb.append([-np.inf])
    gub.append([0])

    return g, glb, gub

# CoM Kinematics Constraints, Currently un-rotated polytope
# In the form of K(CoM - P) <= k
# ContactFrameOrientation: orientation of the contact frame, we need the inverse of this transformation to transform the world frame quantties into the local contact frame
# and then apply the polytopical constraints in the contact frame without rotation


def CoM_Kinematics(SwingLegIndicator=None, CoM_k=None, P=None, K_polytope=None, k_polytope=None,
                   ContactFrameOrientation=ca.DM(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
                   g=None, glb=None, gub=None):

    #ContactFrameOrientation = ca.DM(np.array([[1.0, 0.0, 0.0], [0.0, 1.0 ,0.0], [0.0, 0.0, 1.0]]))
    if SwingLegIndicator == None:
        g.append(K_polytope@ca.inv(ContactFrameOrientation)@(CoM_k-P)-ca.DM(k_polytope))
        glb.append(np.full((len(k_polytope),), -np.inf))
        gub.append(np.full((len(k_polytope),), 0.0))
    else:
        g.append(ca.if_else(SwingLegIndicator, K_polytope@ca.inv(ContactFrameOrientation)@(CoM_k-P)-ca.DM(k_polytope), np.full((len(k_polytope),), -1)))
        glb.append(np.full((len(k_polytope),), -np.inf))
        gub.append(np.full((len(k_polytope),), 0.0))

    return g, glb, gub

# In a very simple form: h_min <= CoM_z - foot_step_z <= h_max


def CoM_to_Foot_Height_Limit(SwingLegIndicator=None, CoM_k=None, P=None, h_min=None, h_max=None,
                             ContactFrameOrientation=ca.DM(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
                             g=None, glb=None, gub=None):
    
    ContactFrameOrientation = ca.DM(np.array([[1.0, 0.0, 0.0], [0.0, 1.0 ,0.0], [0.0, 0.0, 1.0]]))
    if SwingLegIndicator == None:
        # Vector from CoM to footstep
        convertedVec = ca.inv(ContactFrameOrientation)@(CoM_k-P)
        g.append(convertedVec[2])
        glb.append(np.array([h_min]))
        gub.append(np.array([h_max]))
    else:
        # Vector from CoM to footstep
        convertedVec = ca.inv(ContactFrameOrientation)@(CoM_k-P)
        g.append(ca.if_else(SwingLegIndicator,
                 convertedVec[2], np.array([h_min+0.05])))
        glb.append(np.array([h_min]))
        gub.append(np.array([h_max]))

    # if SwingLegIndicator == None:
    #     g.append(CoM_k[2]-P[2])
    #     glb.append(np.array([h_min]))
    #     gub.append(np.array([h_max]))
    # else:
    #     g.append(ca.if_else(SwingLegIndicator, CoM_k[2]-P[2], np.array([h_min+0.05])))
    #     glb.append(np.array([h_min]))
    #     gub.append(np.array([h_max]))

    return g, glb, gub

# Relative Foot Kinematics Constraint, currently non-rotated polytope
# In the form of Q(p_next-p_cur) <= q
# ContactFrameOrientation: orientation of the contact frame, we need the inverse of this transformation to transform the world frame quantties into the local contact frame
# and then apply the polytopical constraints in the contact frame without rotation


def Relative_Foot_Kinematics(SwingLegIndicator=None, p_next=None, p_cur=None, Q_polytope=None, q_polytope=None,
                             ContactFrameOrientation=ca.DM(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])),
                             g=None, glb=None, gub=None):
    
    #ContactFrameOrientation = ca.DM(np.array([[1.0, 0.0, 0.0], [0.0, 1.0 ,0.0], [0.0, 0.0, 1.0]]))
    if SwingLegIndicator == None:
        g.append(Q_polytope@ca.inv(ContactFrameOrientation)@(p_next-p_cur) - ca.DM(q_polytope))
        glb.append(np.full((len(q_polytope),), -np.inf))
        gub.append(np.full((len(q_polytope),), 0.0))
    else:
        g.append(ca.if_else(SwingLegIndicator, Q_polytope@ca.inv(ContactFrameOrientation)@(p_next-p_cur)-ca.DM(q_polytope), np.full((len(q_polytope),), -1)))
        glb.append(np.full((len(q_polytope),), -np.inf))
        gub.append(np.full((len(q_polytope),), 0.0))

    return g, glb, gub


# Constaint on putting CoM in the inner part of the stance foot (swing phase)
# Only put constraint on y axis
def Single_Support_CoM_to_Stance_Center_Limit(SwingLegIndicator=None, CoM_k = None, P_stance = None, min_dist = 0.01, g = None, glb = None, gub = None):
    #we only apply this contraints for y axis for now
    SwingLegIndicator = None
    if not (SwingLegIndicator == None): #if not None, then we apply the constraints, otherwise we do nothing
        g.append(ca.if_else(SwingLegIndicator, ca.fabs(CoM_k[1] - P_stance[1]) - min_dist, np.array([1])))
        glb.append(np.array([0]))
        gub.append(np.array([np.inf]))

    return g, glb, gub

# Footstep location Constraint

#margin for uneven terrain is 0.02
def Stay_on_Surf(P=None, P_TangentX=None, P_TangentY=None, ineq_K=None, ineq_k=None, eq_E=None, eq_e=None, footlength = 0.22, footwidth = 0.12, margin = 0.0, g=None, glb=None, gub=None): 
    # FootStep Location Constraint (On the Patch)
    # Enumperation of each contact point
    # P3----------------P1
    # |                  |
    # |                  |
    # |                  |
    # P4----------------P2
    # margin is to set the safety margin towards the border of the patch, but there is no margin set for the contact points towrads the left edge (this is for making smaller steps)

    #print("Half Foot Length (Bigger Size for kinematics check): ", footlength/2.0)
    #print("Half Foot Width (Bigger Size for kinematics check): ", footwidth/2.0)

    # Contact Points
    P1 = P + footlength/2.0*P_TangentX + footwidth/2.0*P_TangentY
    P2 = P + footlength/2.0*P_TangentX - footwidth/2.0*P_TangentY
    P3 = P - footlength/2.0*P_TangentX + footwidth/2.0*P_TangentY
    P4 = P - footlength/2.0*P_TangentX - footwidth/2.0*P_TangentY

    #-----------On Sufrace Constraint (Inequality integerated the margin from the foot border to the surface borders)
    # Contact Point 1
    # Inequality
    g.append(ineq_K @ P1 - ineq_k + ca.DM(np.array([[margin],[margin],[margin],[margin]])))
    glb.append(np.full((4,), -np.inf))
    gub.append(np.full((4,), 0.0))
    # Equality
    g.append(eq_E.T @ P1 - eq_e)
    glb.append(np.array([0.0]))
    gub.append(np.array([0.0]))

    # Contact Point 2
    # Inequality
    g.append(ineq_K @ P2 - ineq_k + ca.DM(np.array([[margin],[margin],[margin],[margin]])))
    glb.append(np.full((4,), -np.inf))
    gub.append(np.full((4,), 0.0))
    # Equality
    g.append(eq_E.T @ P2 - eq_e)
    glb.append(np.array([0.0]))
    gub.append(np.array([0.0]))

    # Contact Point 3
    # Inequality
    g.append(ineq_K @ P3 - ineq_k + ca.DM(np.array([[margin],[margin],[margin],[margin]])))
    glb.append(np.full((4,), -np.inf))
    gub.append(np.full((4,), 0.0))
    # Equality
    g.append(eq_E.T @ P3 - eq_e)
    glb.append(np.array([0.0]))
    gub.append(np.array([0.0]))

    # Contact Point 4
    # Inequality
    g.append(ineq_K @ P4 - ineq_k + ca.DM(np.array([[margin],[margin],[margin],[margin]])))
    glb.append(np.full((4,), -np.inf))
    gub.append(np.full((4,), 0.0))
    # Equality
    g.append(eq_E.T @ P4 - eq_e)
    glb.append(np.array([0.0]))
    gub.append(np.array([0.0]))

    return g, glb, gub

# Angular Momentum Rate Constraint for Double Support Phase
#  Equation Checked
# Definition of Contact Points of a foot
# P3----------------P1
# |                  |
# |                  |
# |                  |
# P4----------------P2


def Angular_Momentum_Rate_DoubleSupport(SwingLegIndicator=None, Ldot_k=None, CoM_k=None, PL=None, PL_TangentX=None, PL_TangentY=None, PR=None, PR_TangentX=None, PR_TangentY=None, FL1_k=None, FL2_k=None, FL3_k=None, FL4_k=None, FR1_k=None, FR2_k=None, FR3_k=None, FR4_k=None, footlength = 0.2, footwidth = 0.1, g=None, glb=None, gub=None):
    if SwingLegIndicator == None:
        g.append(Ldot_k - (ca.cross((PL + footlength/2.0*PL_TangentX + footwidth/2.0*PL_TangentY - CoM_k), FL1_k) +
                           ca.cross((PL + footlength/2.0*PL_TangentX - footwidth/2.0*PL_TangentY - CoM_k), FL2_k) +
                           ca.cross((PL - footlength/2.0*PL_TangentX + footwidth/2.0*PL_TangentY - CoM_k), FL3_k) +
                           ca.cross((PL - footlength/2.0*PL_TangentX - footwidth/2.0*PL_TangentY - CoM_k), FL4_k) +
                           ca.cross((PR + footlength/2.0*PR_TangentX + footwidth/2.0*PR_TangentY - CoM_k), FR1_k) +
                           ca.cross((PR + footlength/2.0*PR_TangentX - footwidth/2.0*PR_TangentY - CoM_k), FR2_k) +
                           ca.cross((PR - footlength/2.0*PR_TangentX + footwidth/2.0*PR_TangentY - CoM_k), FR3_k) +
                           ca.cross((PR - footlength/2.0*PR_TangentX - footwidth/2.0*PR_TangentY - CoM_k), FR4_k)))
        glb.append(np.array([0.0, 0.0, 0.0]))
        gub.append(np.array([0.0, 0.0, 0.0]))

    else:
        g.append(ca.if_else(SwingLegIndicator, Ldot_k - (ca.cross((PL + footlength/2.0*PL_TangentX + footwidth/2.0*PL_TangentY - CoM_k), FL1_k) +
                                                         ca.cross((PL + footlength/2.0*PL_TangentX - footwidth/2.0*PL_TangentY - CoM_k), FL2_k) +
                                                         ca.cross((PL - footlength/2.0*PL_TangentX + footwidth/2.0*PL_TangentY - CoM_k), FL3_k) +
                                                         ca.cross((PL - footlength/2.0*PL_TangentX - footwidth/2.0*PL_TangentY - CoM_k), FL4_k) +
                                                         ca.cross((PR + footlength/2.0*PR_TangentX + footwidth/2.0*PR_TangentY - CoM_k), FR1_k) +
                                                         ca.cross((PR + footlength/2.0*PR_TangentX - footwidth/2.0*PR_TangentY - CoM_k), FR2_k) +
                                                         ca.cross((PR - footlength/2.0*PR_TangentX + footwidth/2.0*PR_TangentY - CoM_k), FR3_k) +
                                                         ca.cross((PR - footlength/2.0*PR_TangentX - footwidth/2.0*PR_TangentY - CoM_k), FR4_k)), np.array([0.0, 0.0, 0.0])))
        glb.append(np.array([0.0, 0.0, 0.0]))
        gub.append(np.array([0.0, 0.0, 0.0]))

    return g, glb, gub

# Angular Momentum Rate Constraint for Single Support Phase
#   Equation Checked


def Angular_Momentum_Rate_Swing(SwingLegIndicator=None, Ldot_k=None, P=None, P_TangentX=None, P_TangentY=None, CoM_k=None, F1_k=None, F2_k=None, F3_k=None, F4_k=None, footlength = 0.2, footwidth = 0.1, g=None, glb=None, gub=None):

    if SwingLegIndicator == None:

        g.append(Ldot_k - (ca.cross((P + footlength/2.0*P_TangentX + footwidth/2.0*P_TangentY - CoM_k), F1_k) +
                           ca.cross((P + footlength/2.0*P_TangentX - footwidth/2.0*P_TangentY - CoM_k), F2_k) +
                           ca.cross((P - footlength/2.0*P_TangentX + footwidth/2.0*P_TangentY - CoM_k), F3_k) +
                           ca.cross((P - footlength/2.0*P_TangentX - footwidth/2.0*P_TangentY - CoM_k), F4_k)))

        glb.append(np.array([0.0, 0.0, 0.0]))
        gub.append(np.array([0.0, 0.0, 0.0]))
    else:
        g.append(ca.if_else(SwingLegIndicator, Ldot_k - (ca.cross((P + footlength/2.0*P_TangentX + footwidth/2.0*P_TangentY - CoM_k), F1_k) +
                                                         ca.cross((P + footlength/2.0*P_TangentX - footwidth/2.0*P_TangentY - CoM_k), F2_k) +
                                                         ca.cross((P - footlength/2.0*P_TangentX + footwidth/2.0*P_TangentY - CoM_k), F3_k) +
                                                         ca.cross((P - footlength/2.0*P_TangentX - footwidth/2.0*P_TangentY - CoM_k), F4_k)), np.array([0.0, 0.0, 0.0])))
        glb.append(np.array([0.0, 0.0, 0.0]))
        gub.append(np.array([0.0, 0.0, 0.0]))

    return g, glb, gub

# First order Integrator, currently use Euler Integration


def First_Order_Integrator(next_state=None, cur_state=None, cur_derivative=None, h=None, g=None, glb=None, gub=None):
    g.append(next_state - cur_state - h*cur_derivative)
    glb.append(np.array([0.0]))
    gub.append(np.array([0.0]))

    return g, glb, gub

# Unilateral Constraint


def Unilateral_Constraints(SwingLegIndicator=None, F_k=None, TerrainNorm=None, g=None, glb=None, gub=None):

    if SwingLegIndicator == None:  # For Initial Double Support
        g.append(F_k.T@TerrainNorm)
        glb.append(np.array([0.0]))
        gub.append([np.inf])
    else:
        # Activating and de-activating depending on the SwingLegIndicator
        g.append(ca.if_else(SwingLegIndicator, F_k.T@TerrainNorm, np.array([1])))
        glb.append(np.array([0.0]))
        gub.append([np.inf])

    return g, glb, gub

# Zero Forces


def ZeroForces(SwingLegIndicator=None, F_k=None, g=None, glb=None, gub=None):
    g.append(ca.if_else(SwingLegIndicator, F_k, np.array([0.0, 0.0, 0.0])))
    glb.append(np.array([0.0, 0.0, 0.0]))
    gub.append(np.array([0.0, 0.0, 0.0]))

    return g, glb, gub


def FrictionCone(SwingLegIndicator=None, F_k=None, TerrainTangentX=None, TerrainTangentY=None, TerrainNorm=None, miu=None, g=None, glb=None, gub=None):

    if SwingLegIndicator == None:
        # For Initial Phase
        # Friction Cone x-axis Set 1
        g.append(F_k.T@TerrainTangentX - miu*F_k.T@TerrainNorm)
        glb.append([-np.inf])
        gub.append(np.array([0.0]))

        # Friction Cone x-axis Set 2
        g.append(F_k.T@TerrainTangentX + miu*F_k.T@TerrainNorm)
        glb.append(np.array([0.0]))
        gub.append([np.inf])

        # Friction Cone y-axis Set 1
        g.append(F_k.T@TerrainTangentY - miu*F_k.T@TerrainNorm)
        glb.append([-np.inf])
        gub.append(np.array([0.0]))

        # Friction Cone y-axis Set 2
        g.append(F_k.T@TerrainTangentY + miu*F_k.T@TerrainNorm)
        glb.append(np.array([0.0]))
        gub.append([np.inf])

    else:
        # Activate based on the SwingLegIndicator
        # Friction Cone x-axis Set 1
        g.append(ca.if_else(SwingLegIndicator, F_k.T@TerrainTangentX - miu*F_k.T@TerrainNorm, np.array([-1.0])))
        glb.append([-np.inf])
        gub.append(np.array([0.0]))

        # Friction Cone x-axis Set 2
        g.append(ca.if_else(SwingLegIndicator, F_k.T@TerrainTangentX + miu*F_k.T@TerrainNorm, np.array([1.0])))
        glb.append(np.array([0.0]))
        gub.append([np.inf])

        # Friction Cone y-axis Set 1
        g.append(ca.if_else(SwingLegIndicator, F_k.T@TerrainTangentY - miu*F_k.T@TerrainNorm, np.array([-1.0])))
        glb.append([-np.inf])
        gub.append(np.array([0.0]))

        # Friction Cone y-axis Set 2
        g.append(ca.if_else(SwingLegIndicator, F_k.T@TerrainTangentY + miu*F_k.T@TerrainNorm, np.array([1.0])))
        glb.append(np.array([0.0]))
        gub.append([np.inf])

    return g, glb, gub

# Ponton's Convexfication Constraint
#   l_length is the max length of the leg (for normalisation)
#   f_length is the max length of the force vector (for normalisation)
#   P_name: 1) Contact1, Contact2, Contact3, Contact4; 2) Center


def Ponton_Concex_Constraint(SwingLegIndicator=None, P_name=None, P=None, P_TangentX=None, P_TangentY=None, CoM_k=None, l_length=None, f=None, f_length=None, x_p_bar=None, x_q_bar=None, y_p_bar=None, y_q_bar=None, z_p_bar=None, z_q_bar=None, footlength = 0.2, footwidth = 0.1, g=None, glb=None, gub=None):

    # Define Lever Arm based on Contact Location name
    # Definition of Contact Points of a foot NOTE: Different from Contact Patch Definition
    # P3----------------P1
    # |                  |
    # |                  |
    # |                  |
    # P4----------------P2
    if P_name == "Contact1":
        l = P + footlength/2.0*P_TangentX + footwidth/2.0*P_TangentY - CoM_k
    elif P_name == "Contact2":
        l = P + footlength/2.0*P_TangentX - footwidth/2.0*P_TangentY - CoM_k
    elif P_name == "Contact3":
        l = P - footlength/2.0*P_TangentX + footwidth/2.0*P_TangentY - CoM_k
    elif P_name == "Contact4":
        l = P - footlength/2.0*P_TangentX - footwidth/2.0*P_TangentY - CoM_k
    elif P_name == "Center":
        l = P - CoM_k
    else:
        raise Exception("Undefine Contact Point Name")

    a_cvx = np.array([-l[2]/l_length, l[1]/l_length])
    d_cvx = np.array([f[1]/f_length, f[2]/f_length])
    b_cvx = np.array([l[2]/l_length, -l[0]/l_length])
    e_cvx = np.array([f[0]/f_length, f[2]/f_length])
    c_cvx = np.array([-l[1]/l_length, l[0]/l_length])
    f_cvx = np.array([f[0]/f_length, f[1]/f_length])

    x_p = a_cvx + d_cvx
    x_q = a_cvx - d_cvx

    y_p = b_cvx + e_cvx
    y_q = b_cvx - e_cvx

    z_p = c_cvx + f_cvx
    z_q = c_cvx - f_cvx

    g.append(ca.if_else(SwingLegIndicator, x_p_bar-x_p@x_p, np.array([1])))
    glb.append(np.array([0]))
    gub.append(np.array([np.inf]))

    g.append(ca.if_else(SwingLegIndicator, x_q_bar-x_q@x_q, np.array([1])))
    glb.append(np.array([0]))
    gub.append(np.array([np.inf]))

    g.append(ca.if_else(SwingLegIndicator, y_p_bar-y_p@y_p, np.array([1])))
    glb.append(np.array([0]))
    gub.append(np.array([np.inf]))

    g.append(ca.if_else(SwingLegIndicator, y_q_bar-y_q@y_q, np.array([1])))
    glb.append(np.array([0]))
    gub.append(np.array([np.inf]))

    g.append(ca.if_else(SwingLegIndicator, z_p_bar-z_p@z_p, np.array([1])))
    glb.append(np.array([0]))
    gub.append(np.array([np.inf]))

    g.append(ca.if_else(SwingLegIndicator, z_q_bar-z_q@z_q, np.array([1])))
    glb.append(np.array([0]))
    gub.append(np.array([np.inf]))

    return g, glb, gub
