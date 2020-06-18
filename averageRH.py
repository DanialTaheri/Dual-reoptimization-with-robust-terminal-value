# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:18:24 2019
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
import collections

# initialize the random variable and use perfect info. function to 
# find the actions optimal for each time steps
def read_parameters_form_json(file_name):
    with open(file_name) as f:
        config = json.load(f)
    return config

def solveAverageRH(upper_bound, W, l_0, q_0, c_0, numProcess):
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
    model_type = config['model_type']
    initial_cap = config['maxcapacity']
    lag = 0
    # number of inner samples
    K2=config['inner_Samples']
    ARH=open("./Results/DRH-%s-I%s-K%s-K2%s-cap%s.txt"%(model_type, I, K, K2, initial_cap), "wt")
    ARH_action=open("./Results/DRH_actions-%s-I%s-K%s-K2%s-cap%s.txt"%(model_type, I, K, K2,\
                                                                       initial_cap), "wt")
    ARH.write('Average Reoptimization Heuristic\n')
    ARH_action.write('Actions\n')
    

    ARH.write('Number of samples: {} \n'. format (K))
    ARH.write('Number of inner samples: {} \n'. format(K2))
    ARH.write('Time periods: {} \n'.format(I))
    XAvg0=np.zeros(5)
    initial_condition = {'Y0': config['Y0'], 'X0': config['X0'],\
                         'Inflow0': config['Inflow0'], 'Start_time': 0, 'Gamma0': config['plant_cond0']}
    Samples = ScenarioGeneratio.generate_Sample_Avr()

    W2= Samples.generate_Sample(I+1, K2, initial_condition)

    if config["model_type"] == 'noTerminal':
        DRH_MDP = FullMDP.solveMDP(I, K2, W2, l_0, q_0, c_0, numProcess)
    elif config["model_type"] in ['robust', 'nominal']:
        DRH_MDP = FullMDP_Terminal.solveMDP(I, K2, W2, l_0, q_0, c_0, numProcess)
    else:
        raise Error
    
    UB, XAvg0 = DRH_MDP.output()
    XAvg= np.zeros(5)
    XAvg2= np.zeros(5)
    # MDP lowerbound
    lb= np.zeros(K)
    lb_Total = np.zeros(K)
    Price0 = W[0, 0, 0] * W[0, 0, 1] * W[0, 0, 2]
    Inflow0 = W[0, 0, 3] * W[0, 0, 4]
    c_curr0 = W[0, 0, 5] 
    
    RevP = Price0* EnergyCoeff *np.power(disc_fac, 0)* XAvg0[0]
    Cost = C_U*np.power(disc_fac, 0)* XAvg0[1] + C_R*np.power(disc_fac, 0)* XAvg0[3] + XAvg0[4]* C_F *np.power(disc_fac, 0)

    
    # Store policies
    # Generation
    Gen_Policy = np.zeros((I, K))
    #Upgrade, Spill, refurbishment, failure
    Inv_Policy = np.zeros((I, K, 4))
    State_policy = np.zeros((I, K, 1))
    for k in range(K):
        lb[k] = RevP - Cost 
    for k in range(K):
        ARH_action.write('Stage {} \n'. format('0'))
        ARH_action.write('l_init: {}, q_init: {}, c_init: {}\n'.format(l_0, q_0, c_0))
        ARH_action.write('Gen: {}, Up: {}, Sp: {}, Ref: {} \n'. format(
            XAvg0[0], XAvg0[1], XAvg0[2], XAvg0[3] ))    
        ARH_action.write('Sample: {} | Period: {}| Revenue : {}|Cost : {}\n'. format(
                                                                            k, 0, RevP, Cost))
        ARH_action.write('Inflow: {}, c_curr: {}\n'.format(Inflow0, c_curr0)) 
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
            XAvg[m]= XAvg0[m] 
            
        Gen_Policy[0, k] = XAvg[0]
        Inv_Policy[0, k, 0] = XAvg[1]
        Inv_Policy[0, k, 1] = XAvg[2]
        Inv_Policy[0, k, 2] = XAvg[3]
        Inv_Policy[0, k, 3] = XAvg[4]
        State_policy[0, k, 0] = l_new
        
        

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
            W2 = Samples.generate_Sample(I-i+1, K2, initial_condition)
             
            
            if c_curr + c_init >= 1.0:
                ARH_action.write('l_init: {}, q_init: {}, c_init: {}\n'.format(min(l_init, L_max), q_init, c_init))
                ARH_action.write('Gen: {}, Up: {}, Sp: {}, Ref: {}, Flr: {} \n'. format(
                        0, 0, 0, 0, 1 )) 
                ARH_action.write('Inflow: {}, c_curr: {}\n'.format(Inflow, c_curr))  
                lag = T_F_down
                Cost = C_F *np.power(disc_fac, i)
                lb[k] += -Cost
                l_new= max(l_init + Inflow, L_max)
                q_new= q_init 
                
                l_init = l_new
                q_init = q_new
                c_init = 0
                
                XAvg2[0]=0
                XAvg2[1]=0
                XAvg2[2]= 0
                XAvg2[3]= 0
                XAvg2[4]= 1
               #pdb.set_trace()
                for m in range(5):
                    XAvg[m]= XAvg2[m] 
                
                ARH_action.write('l_new: {}, q_new: {}, c_new: {}\n'.format(l_init, q_init, c_init))   
                
                ARH_action.write('Sample: {} | Period: {}| Revenue : {}|Cost : {}\n'. format(
                            k, i, 0, Cost))
                
            else:    

                    
                if lag==0:
                    if config["model_type"] == 'noTerminal':
                        ARH_MDP = FullMDP.solveMDP(I-i, K2, W2, min(l_init, L_max), q_init,\
                                                   c_init, numProcess)
                    elif config["model_type"] in ['robust', 'nominal']:
                        ARH_MDP = FullMDP_Terminal.solveMDP(I-i, K2, W2, min(l_init, L_max),\
                                                            q_init, c_init, numProcess)
                    else:
                        raise Error
                        
                    ub1, XAvg2 = ARH_MDP.output()

                    for m in range(5):
                        XAvg[m]= XAvg2[m]
                    if l_init > L_max:
                        XAvg[2] += l_init - L_max 

                    ARH_action.write('l_init: {}, q_init: {}, c_init: {}\n'.format(min(l_init, L_max), q_init, c_init))
                    ARH_action.write('Gen: {}, Up: {}, Sp: {}, Ref: {}, Flr: {} \n'. format(
                       XAvg[0], XAvg[1], XAvg[2], XAvg[3], XAvg[4]))               
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

                    RevP=Price * EnergyCoeff * np.power(disc_fac, i)* XAvg[0]



                    Cost= C_U*np.power(disc_fac, i)* XAvg[1] + C_R * np.power(disc_fac,i)* XAvg[3] +\
                    XAvg[4]* C_F *np.power(disc_fac, i)

                    lb[k]+= -Cost + RevP 

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
                    ARH_action.write('Gen: {}, Up: {}, Sp: {}, Ref: {}, Flr: {} \n'. format(
                       XAvg[0], XAvg[1], XAvg[2], XAvg[3], XAvg[4] ))                 
                    ARH_action.write('Inflow: {}, c_curr: {}\n'.format(Inflow, c_curr))  

                    #update the endogenous state
                    l_new=  l_init - XAvg[0] + Inflow - XAvg[2]
                    q_new= q_init + XAvg[1]
                    c_new=0


                    l_init=l_new
                    q_init=q_new
                    c_init=c_new


                    RevP = Price * EnergyCoeff * np.power(disc_fac, i)* XAvg[0]


                    Cost= C_U * np.power(disc_fac, i)* XAvg[1] + C_R * np.power(disc_fac,i)* XAvg[3] +\
                    XAvg[4]* C_F *np.power(disc_fac, i)
                    lb[k]+= -Cost + RevP


                    ARH_action.write('l_new: {}, q_new: {}, c_new: {}\n'.format(l_init, q_init, c_init)) 
                    ARH_action.write('Sample: {} | Period: {}| Revenue : {}|Cost : {}\n'. format(
                                            k, i, RevP, Cost))

                    lag-=1
                    
            Gen_Policy[i, k] = XAvg[0]
            Inv_Policy[i, k, 0] = XAvg[1]
            Inv_Policy[i, k, 1] = XAvg[2]
            Inv_Policy[i, k, 2] = XAvg[3]
            Inv_Policy[i, k, 3] = XAvg[4]
            State_policy[i, k, 0] = l_new
        
        ####################################################################
        # Calculating the profit in the Robust part
        ####################################################################
        nb_features, nb_bins = 4, 2
        beta = np.zeros((nb_features, nb_bins))
        threshold = 0.63
        if model_type == 'robust':
            if W[I, k, 2] < 25:
                beta[0, :] = [1.5543, 0.585]
                beta[1, :] = [-2355202.905, -8124486.412]
                beta[2, :] = [103.418, 44.1719]
                beta[3, :] = [460480.896, 6746857.459]
            elif 25 <= W[I, k, 2] < 40:
                beta[0, :] = [3.272, 1.355]
                beta[1, :] = [-4197748.55, -14395691]
                beta[2, :] = [181.702, 77.205]
                beta[3, :] = [788321, 11931227]
            elif 40 <= W[I, k, 2] < 55:
                beta[0, :] = [5.197, 2.1657]
                beta[1, :] = [-6487780.282, -22166358.884]
                beta[2, :] = [279.166, 117.645]
                beta[3, :] = [1215310, 18383787]
            elif 55 <= W[I, k, 2] <= 70:
                beta[0, :] = [7.816748016329668, 3.3523]
                beta[1, :] = [-9348774.127, -31900687.36]
                beta[2, :] = [400.979, 168.6723]
                beta[3, :] = [1726348, 26437073]
            else:
                beta[0, :] = [14.3146, 6.2848]
                beta[1, :] =  [-16522199.781, -56264542.861]
                beta[2, :] = [705.426, 295.418]
                beta[3, :] = [705.426, 46618850]
        if c_new > threshold:
            nb_bin = 1
        else:
            nb_bin = 0
        
        lb_Total[k] = lb[k] + np.power(disc_fac, I)*(l_new * beta[0, nb_bin]+\
                                                     c_new * beta[1, nb_bin]+\
                                                     q_new * beta[2, nb_bin] + beta[3, nb_bin])
#         ARH.write('Sample: {}, Profit: {} \n'. format(k, 
#                 lb[k]))
        ARH.write('Sample: {}, Profit: {} \n'. format(k, 
                lb_Total[k]))
        
    Table_policies = collections.defaultdict(list)
    # for every element in the policies: gen., upgrade, ...
    # we have list of list [[1, ..., i], [1,..., i], ...]
    for k in range(K):
        Table_policies['Generation'].append(list(Gen_Policy[:,k]))
        Table_policies['Upgrade'].append(list(Inv_Policy[:,k, 0]))
        Table_policies['Spill'].append(list(Inv_Policy[:,k, 1]))
        Table_policies['Refurbishment'].append(list(Inv_Policy[:,k, 2]))
        Table_policies['Failure'].append(list(Inv_Policy[:,k, 3]))
        Table_policies['Reservior_level'].append(list(State_policy[:, k, 0]))
    Table_res = './Results/' + "DRH_policy-%s-I%s-K%s-K2%s-cap%s.json" %(model_type, I, K, K2,\
                                                                        initial_cap)
    with open(Table_res, "w") as file:
        json.dump(Table_policies, file)
    
    #######################################
    #Error of MDP
    #######################################
        
    error= np.max(lb)-np.min(lb)
    lower_bound=0
    lower_bound=np.sum(lb)/len(lb)
    error=0
    for k in range(K):
        error+=np.power(lb[k]-lower_bound,2) 
    std_dev=math.sqrt(error/K)
    std_error= std_dev/ math.sqrt(K)
    std_error_per_mean= std_error/lower_bound
    #######################################
    #Error of MDP + Terminal
    #######################################
    error_Total= np.max(lb_Total)-np.min(lb_Total)
    lower_bound_Total=0
    lower_bound_Total=np.sum(lb_Total)/len(lb_Total)
    error_Total=0
    for k in range(K):
        error_Total +=np.power(lb_Total[k]-lower_bound_Total, 2) 
    std_dev_Total=math.sqrt(error_Total/K)
    std_error_Total= std_dev_Total/ math.sqrt(K)
    std_error_per_mean_Total= std_error_Total/lower_bound_Total
    
    end= time.time()
    Time=end - start
    ARH.write('*************************************\n')
    ARH.write('Lower_bound: {} \n'. format(lower_bound_Total))
    ARH.write('*************************************\n')
    ARH.write('std_error: {} \n'. format(std_error_Total)) 
    ARH.write('*************************************\n')
    ARH.write('std_error as percentage of mean: {} \n'. format(std_error_per_mean_Total))  
    #############################################################
    ARH.write('*************************************\n')
    ARH.write('Lower_bound of MDP: {} \n'. format(lower_bound))
    ARH.write('*************************************\n')
    ARH.write('std_error of MDP: {} \n'. format(std_error)) 
    ARH.write('*************************************\n')
    ARH.write('Time: {} \n'. format(time.time()- start))  
    ARH.write('*************************************\n')
    ARH.write('std_error of MDP as percentage of mean: {} \n'. format(std_error_per_mean))  
    ARH.write('*************************************\n')
    Gap = (upper_bound-lower_bound_Total)/ upper_bound
    ARH.write('Gap: {} \n'. format(Gap))  
    ARH.close()
    ARH_action.close()
    return lower_bound





