# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:18:24 2019

@author: smohse3
"""


import numpy as np
import matplotlib.pyplot as plt
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

def solveAverageRH(W, l_0, q_0, c_0, numProcess):
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
    ARH=open("ARH.txt", "wt")
    ARH_action=open("ARH actions.txt", "wt")
    ARH.write('Average Reoptimization Heuristic\n')
    ARH_action.write('Actions\n')
    # number of inner samples
    K2=20
    ARH.write('Number of samples: {} \n'. format (K))
    ARH.write('Number of inner samples: {} \n'. format(K2))
    ARH.write('Time periods: {} \n'.format(I))
    ub1= np.zeros(K2)
    XAvg0=np.zeros(5)
    initial_condition = {'Y0': config['Y0'], 'X0': config['X0'],\
                         'Inflow0': config['Inflow0'], 'Start_time': 0, 'Gamma0': config['plant_cond0']}
    Samples = ScenarioGeneratio.generate_Sample_Avr()

    W2= Samples.generate_Sample(I, K2, initial_condition)

    DRH_MDP = FullMDP.solveMDP(I, K2, W2, l_0, q_0, c_0, numProcess)
    ub1, XAvg0 = DRH_MDP.output()
    XAvg= np.zeros(5)
    XAvg2= np.zeros(5)
    lb= np.zeros(K)
    
    Price0 = W[0, 0, 0] * W[0, 0, 1] * W[0, 0, 2]
    Inflow0 = W[0, 0, 3] * W[0, 0, 4]
    c_curr0 = W[0, 0, 5] 
    

    


    RevP = Price0* EnergyCoeff *np.power(disc_fac, 0)* XAvg0[0]
    Cost= C_U*np.power(disc_fac, 0)* XAvg0[1] + C_R*np.power(disc_fac, 0)* XAvg0[3]
    Cost_F= XAvg0[4]* C_F *np.power(disc_fac, 0)
    
    for k in range(K):
        lb[k] = RevP - Cost - Cost_F 
        # set the endogenous variables equal to initial reservior level, 
        # capacity, and the situation of the plant
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
            

        # for each period
        for i in range(1, I):
            ARH_action.write('Stage {} \n'. format(i))   
            
            if XAvg[4] == 1:
                lag = T_F_down
            if XAvg[3] == 1:
                lag = T_R_down

            
            Price = W[i, k, 0]* W[i, k, 1] * W[i, k, 2]
            Inflow = W[i, k, 3] * W[i, k, 4]
            c_curr = W[i, k, 5]
            
            initial_condition = {'Y0': np.log(W[i, k, 2]), 'X0': np.log(W[i, k, 1]),\
                         'Inflow0': np.log(W[i, k, 4]), 'Start_time': i, 'Gamma0': W[i, k, 5]}
            W2 = Samples.generate_Sample(I-i, K2, initial_condition)
             
            
            if c_curr + c_init >= 1.0:
                ARH_action.write('l_init: {}, q_init: {}, c_init: {}\n'.format(min(l_init, L_max), q_init, c_init))
                ARH_action.write('Gen: {}, Up: {}, Sp: {}, Ref: {}, failure: {} \n'. format(
                        0, 0, 0, 0, 1 )) 
                ARH_action.write('Inflow: {}, c_curr: {}\n'.format(Inflow, c_curr))  
                lag = T_F_down
                lb[k] += -C_F *np.power(disc_fac, i)
                l_new= max(l_init + Inflow, L_max)
                q_new= q_init 
                
                l_init = l_new
                q_init = q_new
                c_init = 0
                
                ARH_action.write('l_new: {}, q_new: {}, c_new: {}\n'.format(l_init, q_init, c_init))   
                
                ARH_action.write('Sample: {} | Period: {}| Revenue : {}|Cost : {}\n'. format(
                            k, i, 0, lb[k]))
                continue

                    
            if lag==0:
                ARH_MDP = FullMDP.solveMDP(I-i, K2, W2, min(l_init, L_max), q_init, c_init, numProcess)
                ub1, XAvg2 = ARH_MDP.output()
                
                for m in range(5):
                    XAvg[m]= XAvg2[m]
                if l_init > L_max:
                    XAvg[2] += l_init - L_max 
                    
                ARH_action.write('l_init: {}, q_init: {}, c_init: {}\n'.format(min(l_init, L_max), q_init, c_init))
                ARH_action.write('Gen: {}, Up: {}, Sp: {}, Ref: {} \n'. format(
                   XAvg[0], XAvg[1], XAvg[2], XAvg[3] ))                 
                ARH_action.write('Inflow: {}, c_curr: {}\n'.format(Inflow, c_curr))  
                l_new= l_init - XAvg[0] + Inflow - XAvg[2]
                q_new= q_init + XAvg[1]
                if XAvg[0] > 0:
                    c_new= c_init + c_curr
                else:
                    c_new= c_init
            
            
                l_init=l_new
                q_init=q_new
                c_init=c_new
                
                ARH_action.write('l_new: {}, q_new: {}, c_new: {}\n'.format(l_init, q_init, c_init))                
                
                RevP=Price * EnergyCoeff * np.power(disc_fac, i-1)* XAvg[0]



                Cost= C_U*np.power(disc_fac, i-1)* XAvg[1] + C_R * np.power(disc_fac,i-1)* XAvg[3]
                Cost_F= XAvg[4]* C_F *np.power(disc_fac, i-1)
                lb[k]+= -Cost + RevP -Cost_F
            
                ARH_action.write('Sample: {} | Period: {}| Revenue : {}|Cost : {}\n'. format(
                        k, i, RevP, Cost))
          
    
            else:
            
                XAvg2[0]=0
                XAvg2[1]=0
                XAvg2[2]= max(0,-L_max+ ( l_init + Inflow))
                XAvg2[3]=0
                XAvg2[4]=0
               #pdb.set_trace()
                for m in range(5):
                    XAvg[m]= XAvg2[m] 
                    
                ARH_action.write('l_init: {}, q_init: {}, c_init: {}\n'.format(min(l_init, L_max), q_init, c_init))
                ARH_action.write('Gen: {}, Up: {}, Sp: {}, Ref: {} \n'. format(
                   XAvg[0], XAvg[1], XAvg[2], XAvg[3] ))                 
                ARH_action.write('Inflow: {}, c_curr: {}\n'.format(Inflow, c_curr))  
                    
                #update the endogenous state
                l_new=  l_init - XAvg[0] + Inflow - XAvg[2]
                q_new= q_init + XAvg[1]
                c_new=0
            
                
                l_init=l_new
                q_init=q_new
                c_init=c_new
                 
                            
                RevP = Price * EnergyCoeff * np.power(disc_fac, i-1)* XAvg[0]
    

                Cost= C_U * np.power(disc_fac, i-1)* XAvg[1] + C_R * np.power(disc_fac,i-1)* XAvg[3]
                Cost_F= XAvg[4]* C_F *np.power(disc_fac, i-1)
                lb[k]+= -Cost + RevP - Cost_F
                

                ARH_action.write('l_new: {}, q_new: {}, c_new: {}\n'.format(l_init, q_init, c_init)) 
                ARH_action.write('Sample: {} | Period: {}| Revenue : {}|Cost : {}\n'. format(
                                        k, i, RevP, Cost))

                lag-=1



        ARH.write('Sample: {}, Profit: {} \n'. format(k, 
                lb[k]))
        
    error= np.max(lb)-np.min(lb)
    lower_bound=0
    lower_bound=np.sum(lb,0)/len(lb)
    error=0
    for k in range(K):
        error+=np.power(lb[k]-lower_bound,2) 
    std_dev=math.sqrt(error/K)
    std_error= std_dev/ math.sqrt(K)
    std_error_per_mean= std_error/lower_bound
    end= time.time()
    Time=end - start
    ARH.write('*************************************\n')
    ARH.write('Lower_bound: {} \n'. format(lower_bound))
    ARH.write('*************************************\n')
    ARH.write('std_error: {} \n'. format(std_error)) 
    ARH.write('*************************************\n')
    ARH.write('Time: {} \n'. format(time.time()- start))  
    ARH.write('*************************************\n')
    ARH.write('std_error as percentage of mean: {} \n'. format(std_error_per_mean))  
    ARH.close()
    ARH_action.close()
    return lower_bound





