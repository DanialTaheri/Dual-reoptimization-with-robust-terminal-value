# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 14:18:24 2019

@author: smohse3
"""


import numpy as np
import matplotlib.pyplot as plt
import inputProcessing
import ScenarioGeneratio
from gurobipy import Model, Var, GRB, quicksum
import PerfectInformation 
import pdb
import time
import math
# initialize the random variable and use perfect info. function to 
# find the actions optimal for each time steps


def solveAverageRH(gp, wp, W, l_0, q_0, c_0, I2):
    T_R_down=2
    T_F_down=4
    lag=0
    L_max=1200
    start=time.time()
    ARH=open("ARH.txt", "wt")
    ARH_action=open("ARH actions.txt", "wt")
    ARH.write('Average Reoptimization Heuristic\n')
    ARH_action.write('Actions\n')
     # number of samples
    K=int(gp[1])
    # number of inner samples
    K2=3
    # time period 
    I=int(gp[0])
    ARH.write('Number of samples: {} \n'. format (K))
    ARH.write('Number of inner samples: {} \n'. format( K2) )
    ARH.write('Time periods: {} \n'.format(I))
    ub1= np.zeros(K2)
    XAvg0=np.zeros(5)

    W2= generate_AR1 (I, K2, wp)

    ub1, XAvg0= solveMDP_robust(gp, I, K2, W2, l_0, q_0, c_0, I2)

    XAvg= np.zeros(5)
    XAvg2= np.zeros(5)


    lb= np.zeros(K)
    for k in range(K):
        
        
        lb[k]=0
    

# set the endogenous variables equal to initial reservior level, 
# capacity, and the situation of the plant
    
        l_init=l_0
        q_init=q_0
        c_init=c_0
        
        for m in range(5):
            XAvg[m]= XAvg0[m]
            
 #update the endogenous state
        l_new= l_init - XAvg[0] + W[0, k, 1]- XAvg[2]
        q_new= q_init + XAvg[1]
        if XAvg[0] > 0:
            c_new= c_init + W[0,k, 2]
        else:
            c_new= c_init
            
            
        l_init=l_new
        q_init=q_new
        c_init=c_new

        
        #For each time period
        lag=0

        for i in range(1, I):
            ARH_action.write('Stage {} \n'. format(i))   
            
            if XAvg[4]==1:
                lag=T_F_down
            if XAvg[3]==1:
                lag=T_R_down
                                    
                    
            if lag==0:
                
                
                RevP=W[i-1, k, 0]*gp[5]*np.power(gp[6], i-1)* XAvg[0]
    
            
                ARH_action.write('Gen: {}, Up: {}, Sp: {}, Ref: {} \n'. format(
                   XAvg[0], XAvg[1], XAvg[2], XAvg[3] )) 

                Cost= gp[7]*np.power(gp[6], i-1)* XAvg[1] + gp[8]*np.power(gp[6],i-1)* XAvg[3]
                Cost_F= XAvg[4]* 100 *np.power(gp[6], i-1)
                lb[k]+= -Cost+RevP -Cost_F
            
                ARH_action.write('Sample: {} | Period: {}| Revenue : {}|Cost : {}\n'. format(
                        k, i-1, RevP, Cost))
          
           
                W2=np.zeros((I-i+1, K2,2))
           
                wp[0,0]= W[i,k,0]
                wp[1,0]= W[i,k,1]
                wp[2,0]= W[i,k,2]
                
                W2= generate_AR1(I-i+1, K2, wp)
                ub1, XAvg2 = solveMDP_robust(gp, I+1-i, K2, W2, l_init, q_init, c_init, I2)
                for m in range(5):
                    XAvg[m]= XAvg2[m]
                    
                l_new= l_init - XAvg[0] + W[i, k, 1]- XAvg[2]
                q_new= q_init + XAvg[1]
                if XAvg[0] > 0:
                    c_new= c_init + W[i, k, 2]
                else:
                    c_new= c_init
            
            
                l_init=l_new
                q_init=q_new
                c_init=c_new
                
                
            else:
                            
                RevP=W[i-1, k, 0]*gp[5]*np.power(gp[6], i-1)* XAvg[0]
    
            
                ARH_action.write('Gen: {}, Up: {}, Sp: {}, Ref: {} \n'. format(
                                    XAvg[0], XAvg[1], XAvg[2], XAvg[3] )) 

                Cost= gp[7]*np.power(gp[6], i-1)* XAvg[1] + gp[8]*np.power(gp[6],i-1)* XAvg[3]
                Cost_F= XAvg[4]* 100 *np.power(gp[6], i-1)
                lb[k]+= -Cost+RevP -Cost_F
            
            
                ARH_action.write('Sample: {} | Period: {}| Revenue : {}|Cost : {}\n'. format(
                                        k, i-1, RevP, Cost))
          
                 
           
                W2=np.zeros((I-i+1, K2,2))
           
                wp[0,0]= W[i,k,0]
                wp[1,0]= W[i,k,1]
                wp[2,0]= W[i,k,2]
            
                W2= generate_AR1(I-i+1, K2, wp)
                XAvg2[0]=0
                XAvg2[1]=0
                XAvg2[2]= max(0,-L_max+ ( l_init + W[i, k, 1]))
                XAvg2[3]=0
                XAvg2[4]=0
               #pdb.set_trace()
                for m in range(5):
                    XAvg[m]= XAvg2[m] 
                    
                #update the endogenous state
                l_new=  l_init - XAvg[0] + W[i, k, 1]- XAvg[2]
                q_new= q_init + XAvg[1]
                c_new=0
            
                
                l_init=l_new
                q_init=q_new
                c_init=c_new
                 
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


    # determine the unit revenue of power, cost of 
 #maintenance, upgrade and failure

 # Update the initial reservior level, condition of the plant and the capacity
 # for each period


 # Sample for RH based on the expected future inflow & price & random plant condition
 
 
 
 
 
 # Solve the perfect information function for remaning periods, 
 # new sample of price, inflow
 # Type the 


