# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:04:36 2019

@author: smohse3
"""

import numpy as np
#import matplotlib.pyplot as plt
import ScenarioGeneratio
from gurobipy import Model, Var, GRB, quicksum
import FullMDP, FullMDP_Terminal
import pdb
import time
import math
import json
# initialize the random variable and use perfect info. function to 
# find the actions optimal for each time steps

def read_parameters_form_json(file_name):
    with open(file_name) as f:
        config = json.load(f)
    return config


def solveRH(W, l_0, q_0, c_0, numProcess):
    start= time.time()
    config = read_parameters_form_json('Parameters.json')
    I = config['nPeriods']
    # number of samples
    K = config['n_Samples']
    L_min = config['minLevel']
    L_max = config['maxLevel']
    T_R_down = config['ref_down']
    T_F_down = config['failure_down']
    C_F = config['CostRf']
    C_R = config['CostRf']
    C_U = config['CostUp']
    disc_fac = config['disc_rate']
    EnergyCoeff = config['EnergyCoeff']
    lag = 0
    ###################################
    # initializing text files
    g=open("RH.txt", "wt")
    h=open("RH actions.txt", "wt")
    g.write('Reoptimization Heuristic\n')
    h.write('Actions\n')
    ####################################
    # number of inner samples
    K2 = 1
    # time period 
    g.write('Number of samples: {} \n'. format (K) )
    g.write('Time periods: {} \n'.format(I))
    ub1= np.zeros(K2)
    # Outer samples
    Average_Samples = ScenarioGeneratio.generate_Sample_Avr()
    initial_condition = {'Y0': config['Y0'], 'X0': config['X0'],\
                         'Inflow0': config['Inflow0'], 'Start_time': 0, 'Gamma0': config['plant_cond0']}
    Samples = ScenarioGeneratio.generate_Sample_Avr()
    W2 = Samples.generate_Average(I+1, K2, initial_condition)
    
    # Optimization at time t=0
    RH_MDP= FullMDP_Terminal.solveMDP(I, K2, W2, l_0, q_0, c_0, numProcess)
    ub1, XAvg0 = RH_MDP.output()
    XAvg= np.zeros(5)
    XAvg2= np.zeros(5)
    lb= np.zeros(K)
    
    Price0 = W[0, 0, 0] * W[0, 0, 1] * W[0, 0, 2]
    Inflow0 = W[0, 0, 3] * W[0, 0, 4]
    c_curr0 = W[0, 0, 5] 

    # Picking the first action of preparing to step forward
    RevP = Price0* EnergyCoeff *np.power(disc_fac, 0)* XAvg0[0]
    Cost= C_U*np.power(disc_fac, 0)* XAvg0[1] + C_R*np.power(disc_fac, 0)* XAvg0[3]
    Cost_F= XAvg0[4]* C_F *np.power(disc_fac, 0)

    # For each outer sample
    for k in range(K):
        # initialize lower bounds with revenue from the action at time t=0
        lb[k] = RevP - Cost - Cost_F 
    

    
    for k in range(K):
        h.write('Stage {} \n'. format('0'))
        h.write('Sample: {} | Period: {}| Revenue : {}|Cost : {}\n'. format(
                                                                            k, 0, RevP, Cost))
        h.write('Inflow: {}, c_curr: {}\n'.format(Inflow0, c_curr0)) 
        #update the endogenous state
        l_new= l_0 - XAvg0[0] + Inflow0 - XAvg0[2]
        q_new= q_0 + XAvg0[1]
        if XAvg0[0] > 0:
            c_new= c_0 + c_curr0
        else:
            c_new= c_0

        l_init=l_new
        q_init=q_new
        c_init=c_new
        
        for m in range(5):
            XAvg[m]= XAvg2[m] 

        #For each time period
        for i in range(1, I):
            h.write('Stage {} \n'. format(i))
            #check if upgrade or redurbishment happened
            if XAvg[4] == 1:
                lag = T_F_down
            if XAvg[3] == 1:
                lag = T_R_down

            
            Price = W[i, k, 0]* W[i, k, 1] * W[i, k, 2]
            Inflow = W[i, k, 3] * W[i, k, 4]
            c_curr = W[i, k, 5]
            
            initial_condition = {'Y0': np.log(W[i, k, 2]), 'X0': np.log(W[i, k, 1]),\
                         'Inflow0': np.log(W[i, k, 4]), 'Start_time': i, 'Gamma0': W[i, k, 5]}
            W2 = Samples.generate_Average(I-i+1, K2, initial_condition)

            # To prevent infeasibility of the opt. b/c of c_init > 1
            if c_curr + c_init >= 1.0:
                ################################################################
                h.write('l_init: {}, q_init: {}, c_init: {}\n'.format(min(l_init, L_max), q_init, c_init))
                h.write('Gen: {}, Up: {}, Sp: {}, Ref: {}, failure: {} \n'. format(
                        0, 0, 0, 0, 1 )) 
                h.write('Inflow: {}, c_curr: {}\n'.format(Inflow, c_curr))   
                ##################################################################
                lag = T_F_down
                lb[k] += -C_F *np.power(disc_fac, i)
                # update endogenous states
                l_new= max(l_init + Inflow, L_max)
                q_new= q_init 
                
                l_init = l_new
                q_init = q_new
                h.write('Sample: {} | Period: {}| Revenue : {}|Cost : {}\n'. format(
                            k, i, 0, lb[k]))
                continue


            #lag captures the last investment deicision which is made
            if lag == 0:  
                # ub1, XAvg2 = solvePerfectInformation(gp, I+1-i, K2, W2, l_init, q_init, c_init)
                RH_MDP = FullMDP_Terminal.solveMDP(I-i, K2, W2, min(l_init, L_max), q_init, c_init, numProcess)
                ub1, XAvg2 = RH_MDP.output()
            
                for m in range(5):
                    XAvg[m]= XAvg2[m] 
                if l_init > L_max:
                    XAvg[2] += l_init - L_max
                    
                h.write('l_init: {}, q_init: {}, c_init: {}\n'.format(min(l_init, L_max), q_init, c_init))
                h.write('Gen: {}, Up: {}, Sp: {}, Ref: {} \n'. format(
                        XAvg[0], XAvg[1], XAvg[2], XAvg[3] )) 
                h.write('Inflow: {}, c_curr: {}\n'.format(Inflow, c_curr))                
                l_new= l_init - XAvg[0] + Inflow - XAvg[2]
                q_new= q_init + XAvg[1]
                if XAvg[0] > 0:
                    c_new= c_init + c_curr
                else:
                    c_new= c_init
            
            
                l_init=l_new
                q_init=q_new
                c_init=c_new

                h.write('l_new: {}, q_new: {}, c_new: {}\n'.format(l_new, q_new, c_new))
                
                # Calculating costs and revenue
                RevP = Price* EnergyCoeff *np.power(disc_fac, i)* XAvg[0]
                Cost= C_U*np.power(disc_fac, i)* XAvg[1] + C_R*np.power(disc_fac,i)* XAvg[3]
                Cost_F= XAvg[4]* C_F *np.power(disc_fac, i)
                lb[k]+= -Cost + RevP - Cost_F
            
                h.write('Sample: {} | Period: {}| Revenue : {}|Cost : {}\n'. format(
                            k, i, RevP, Cost))
                
                

            # If lag is nonzero then the production is zero
            else:
                XAvg2[0]=0
                XAvg2[1]=0
                XAvg2[2]= max(0,-L_max+ ( l_init + Inflow))
                XAvg2[3]=0
                XAvg2[4]=0
                
                for m in range(5):
                    XAvg[m]= XAvg2[m] 
                # For debugging: delete for speed
                #############################################################################
                h.write('l_init: {}, q_init: {}, c_init: {}\n'.format(l_init, q_init, c_init))
                h.write('Gen: {}, Up: {}, Sp: {}, Ref: {} \n'. format(
                        XAvg[0], XAvg[1], XAvg[2], XAvg[3] )) 
                h.write('Inflow: {}, c_curr: {}\n'.format(Inflow, c_curr))   
                #############################################################################
                #update the endogenous state
                l_new =  l_init - XAvg[0] + Inflow - XAvg[2]
                q_new = q_init + XAvg[1]
                c_new = 0
            
                
                l_init = l_new
                q_init = q_new
                c_init = c_new
                
                h.write('l_new: {}, q_new: {}, c_new: {}\n'.format(l_new, q_new, c_new))
                
                # Calculating costs and revenue
                RevP = Price* EnergyCoeff *np.power(disc_fac, i)* XAvg[0]
                Cost= C_U*np.power(disc_fac, i)* XAvg[1] + C_R*np.power(disc_fac,i)* XAvg[3]
                Cost_F= XAvg[4]* C_F *np.power(disc_fac, i)
                
                lb[k]+= -Cost + RevP - Cost_F
                lag-=1

        g.write('Sample: {}, Profit: {} \n'. format(k, 
                lb[k]))
        
    
    lower_bound=0
    lower_bound=np.sum(lb,0)/K
    error=0
    for k in range(K):
        error+=np.power(lb[k]-lower_bound,2) 
    std_dev=math.sqrt(error/K)
    std_error= std_dev/ math.sqrt(K)
    std_error_per_mean= std_error/lower_bound
    end= time.time()
    Time = end - start
    
    g.write('*************************************\n')
    g.write('Lower_bound: {} \n'. format(lower_bound))
    g.write('*************************************\n')
    g.write('std_error: {} \n'. format(std_error)) 
    g.write('*************************************\n')
    #g.write('Time: {} \n'. format(Time))  
    #g.write('*************************************\n')
    g.write('std_error as percentage of mean: {} \n'. format(std_error_per_mean))  
    g.write('*************************************\n')
    g.write("Time: %s seconds ----" %(Time))
    g.close()
    h.close()
    return lower_bound