# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:40:12 2019

@author: smohse3
"""

import numpy as np
import matplotlib.pyplot as plt
import inputProcessing
import ScenarioGeneratio
from gurobipy import Model, Var, GRB, quicksum
import MDP_robust 
import pdb
import averageRH
import RH
import time




wp=inputProcessing.readProcessParams('I_processparametersAR1.txt')
gp=inputProcessing.readProblemParams('I_problemParams.txt')
l_init=100
q_init=50
c_init=0 
W=generate_AR1(gp[0], gp[1], wp)
I2= 180
K=int(gp[1])
start_time = time.time()
ub, XAvg= solveMDP_robust(gp, int(gp[0]), int(gp[1]), W, l_init, q_init, c_init, I2)
Perfectinfo_time = time.time() - start_time
upper_bound= np.sum(ub,0)/len(ub)
print("######################################################")
print("Terminated Perfect Information")

#start_time = time.time()
#lower_bound_ARH= solveAverageRH(gp, wp, W, l_init, q_init, c_init, I2)
#print('Ub : {}, Lb : {}'. format(upper_bound, lower_bound_ARH))
#print((upper_bound-lower_bound_ARH)/ upper_bound)
#DRH_time = time.time() - start_time
#
#start_time = time.time()
#lower_bound_RH= solveRH(gp, wp, W, l_init, q_init, c_init, I2)
#print('Ub : {}, Lb : {}'. format(upper_bound, lower_bound_RH))
#print((upper_bound-lower_bound_RH)/ upper_bound)
#RH_time = time.time() - start_time