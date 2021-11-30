import numpy as np
import casadi as ca

import multicontact_learning_local_objectives.python.ocp_build as ocp_build

a = ca.SX.sym('a')
b = ca.SX.sym('b')
c = ca.SX.sym('c')

d = ca.vertcat(a,b,c)
e = ca.vertcat(c,a,a)
print(d)

g = []
glb = []
gub = []

print(ocp_build.std_eq_constraint(a = d, b = e, g = g, glb = glb, gub = gub))