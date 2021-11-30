#Conver between Representations of Contact Locations

import casadi as ca
import numpy as np

#Convert from 3D Points to Convex Combination Representation, using QP to get the coefficients
#ContactLocation: an one row numpy array
def Point3D_to_ConvexCombination(ContactLocation = None, ContactSurf = None):
    #Creat the QP problem, number of coefficient is aligned with number of vertex of the ContactSurf variable
    NumCoef = len(ContactSurf)
    #   Decision variable and its bounds, each a > 0 and a < 1
    a = ca.SX.sym('a',NumCoef);    a_lb = np.zeros(NumCoef);   a_ub = np.ones(NumCoef)
    #Build the constraint container
    g = [];    glb = [];   gub = []
    #No cost, a fesibility problem only
    J = 0.0

    #Start Building the Constraint
    #   Sigma(a) = 1
    g.append(np.ones((1,NumCoef))@a);     glb.append(np.array([1]));    gub.append(np.array([1]))

    #   Vertex Combination
    g.append(np.transpose(ContactSurf)@a);   glb.append(ContactLocation);   gub.append(ContactLocation)

    #Generate initial seed
    np.random.seed();    a_init = np.random.rand(NumCoef,).flatten()

    #Build Optimization problem
    g = ca.vertcat(*g)
    glb = np.concatenate((glb),axis=None)
    gub = np.concatenate((gub),axis=None)

    qp_prob = {'f': J, 'x': a, 'g': g}
    solver = ca.qpsol('solver', 'qpoases', qp_prob)

    #Solve the problem
    sol = solver(x0=a_init, lbx=a_lb, ubx=a_ub, lbg=glb, ubg=gub)

    if solver.stats()["success"] == False:
        raise Exception("Unable to find correct coefficient")
    elif solver.stats()["success"] == True:
        a_sol = np.array(sol["x"])

    print("Coefficients are: ", a_sol)

    return a_sol

#Convert from Convex Combination to 3D points
def ConvexCombination_to_Point3D(Coef = None, ContactSurf = None):
    #Simple Matrix Multiplication
    P = np.transpose(ContactSurf)@np.reshape(Coef, (4,1))

    return P