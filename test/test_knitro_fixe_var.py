import casadi as ca
import numpy as np
from casadi import *
import pickle

x = ca.SX.sym("x",2)
xlb = [-1,-1]
xub = [1,1]

J = x[0]**2 + x[1]**2

g = x[0] - 1
glb = [0]
gub = [0]

#-----------------------------------------------------------------------------------------------------------------------
#   Build Solvers
prob = {'x': x, 'f': J, 'g': g}
opts = {}

#------------Kintro------------------
#Good Setup of Knitro
opts["knitro.presolve"] = 1
opts["knitro.honorbnds"] = 0
opts["knitro.OutLev"] = 2
opts["knitro.bar_directinterval"] = 0
opts["knitro.maxit"]=10000
opts["knitro.maxtime_real"]=10
opts["knitro.bar_feasible"]=2

solver = ca.nlpsol('solver', 'knitro', prob, opts)


res = solver(x0=[-2,2], lbx = xlb, ubx = xub, lbg = glb, ubg = gub)




