import numpy as np
from ScenarioGeneratio import generate_Sample_Avr
from gurobipy import Model, Var, GRB, quicksum
import FullMDP, FullMDP_Terminal 
import pdb
from averageRH import solveAverageRH
from RH import solveRH
import time
import json
import copyreg
import types

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copyreg.pickle(types.MethodType, _pickle_method)





def read_parameters_form_json(file_name):
    with open(file_name) as f:
        config = json.load(f)
    return config

config = read_parameters_form_json('Parameters.json')
# intiial servior level
l_init = 10000
# initial capacity
q_init = config["maxcapacity"]
# initial plant condition
c_init = config['plant_cond0']
# Generating samples
Samples = generate_Sample_Avr()
initial_condition = {'X0': config['X0'], 'Y0': config['Y0'],\
                    'Inflow0': config['Inflow0'], 'Start_time': 0,\
                    'Gamma0': config['plant_cond0']}
W = Samples.generate_Sample(config['nPeriods']+1, config['n_Samples'], initial_condition)

# Calculating upper bound
start_time = time.time()
if config["model_type"] == 'noTerminal':
    PerfectInformation_MDP = FullMDP.solveMDP(config['nPeriods'], config['n_Samples'], W, l_init,\
                                              q_init, c_init, numProcess=1)
elif config["model_type"] in ['robust', 'nominal']:
    PerfectInformation_MDP = FullMDP_Terminal.solveMDP(config['nPeriods'], config['n_Samples'], W,\
                                                       l_init, q_init, c_init, numProcess=1)
else:
    raise Error
    
ub, XAvg = PerfectInformation_MDP.output()
Perfectinfo_time = time.time() - start_time
upper_bound= ub
print("Perfectinfo_time", Perfectinfo_time)
print("upper_bound :", upper_bound)
print("######################################################")
print("Terminated Perfect Information")

###########################################
# Dual Reoptimization heuristic
#########################################
# start_time = time.time()
# lower_bound_ARH = solveAverageRH(upper_bound, W, l_init, q_init, c_init, numProcess=20)
# DRH_time = time.time() - start_time
# print("DRH_time", DRH_time)


#  Reoptimization heuristic
"""
start_time = time.time()
lower_bound_RH= solveRH(W, l_init, q_init, c_init, numProcess=1)
print('Ub : {}, Lb : {}'. format(upper_bound, lower_bound_RH))
print((upper_bound-lower_bound_RH)/ upper_bound)
RH_time = time.time() - start_time
print("RH_time", RH_time)
print("######################################################")
print("Terminated Perfect Information")
"""
