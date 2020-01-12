# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 20:04:36 2019

@author: smohse3
"""

import numpy as np
import matplotlib.pyplot as plt
from inputProcessing import *
import ScenarioGeneratio
from gurobipy import Model, Var, GRB, quicksum
import MDP_robust 
import pdb
import time
import math
# initialize the random variable and use perfect info. function to 
# find the actions optimal for each time steps

start_time = time.time()
def solveRH(gp, wp, W, l_0, q_0, c_0, I2):
    start= time.time()
    T_R_down=2
    T_F_down=4
    lag=0
    L_max=1200
    g=open("RH.txt", "wt")
    h=open("RH actions.txt", "wt")
    g.write('Reoptimization Heuristic\n')
    h.write('Actions\n')
    # number of samples
    K=int(gp[1])
    # number of inner samples
    K2=1
    # time period 
    I=int(gp[0])
    g.write('Number of samples: {} \n'. format (K) )
    g.write('Time periods: {} \n'.format(I))
    ub1= np.zeros(K2)
    XAvg=[]
    W2= Average_AR1 (I, K2, wp)
    #ub1, XAvg0= solvePerfectInformation(gp, I, K2, W2, l_0, q_0, c_0 )
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
            h.write('Stage {} \n'. format(i))
            #determine the revenue  
            if XAvg[4]==1:
                lag=T_F_down
            if XAvg[3]==1:
                lag=T_R_down
            print("i:", i)
                                    
             #lag captures the last investment deicision which is made
            if lag==0:
                
                    
                RevP=W[i-1, k, 0]*gp[5]*np.power(gp[6], i-1)* XAvg[0]
    
            
                h.write('Gen: {}, Up: {}, Sp: {}, Ref: {} \n'. format(
                        XAvg[0], XAvg[1], XAvg[2], XAvg[3] )) 
#                print('Gen: {}, Up: {}, Sp: {}, Ref: {} \n'. format(
#                        XAvg[0], XAvg[1], XAvg[2], XAvg[3] )) 

                Cost= gp[7]*np.power(gp[6], i-1)* XAvg[1] + gp[8]*np.power(gp[6],i-1)* XAvg[3]
                Cost_F= XAvg[4]* 100 *np.power(gp[6], i-1)
                lb[k]+= -Cost+RevP -Cost_F
            
                h.write('Sample: {} | Period: {}| Revenue : {}|Cost : {}\n'. format(
                            k, i-1, RevP, Cost))
                W2=np.zeros((I-i+1, K2,2))
           
                wp[0,0]= W[i,k,0]
                wp[1,0]= W[i,k,1]
                wp[2,0]= W[i,k,2]
                W2= Average_AR1(I-i+1, K2, wp)
                # ub1, XAvg2 = solvePerfectInformation(gp, I+1-i, K2, W2, l_init, q_init, c_init)
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

                
            # If lag is nonzero then the production is zero
            else:
                RevP=W[i-1, k, 0]*gp[5]*np.power(gp[6], i-1)* XAvg[0]
                h.write('Gen: {}, Up: {}, Sp: {}, Ref: {}, Flr:{}\n'. format(
                                XAvg[0], XAvg[1], XAvg[2], XAvg[3], XAvg[4]))
#                print('Lag0:Gen: {}, Up: {}, Sp: {}, Ref: {} \n'. format(
#                            XAvg[0], XAvg[1], XAvg[2], XAvg[3] )) 

                Cost= gp[7]*np.power(gp[6], i-1)* XAvg[1] + gp[8]*np.power(gp[6],i-1)* XAvg[3]
                Cost_F= XAvg[4]* 100 *np.power(gp[6], i-1)
                lb[k]+= -Cost+RevP -Cost_F
            
                h.write('Sample: {} | Period: {}| Revenue : {}|Cost : {}\n'. format(
                        k, i-1, RevP, Cost))
               

           
                W2=np.zeros((I-i+1, K2,2))
                wp[0,0]= W[i,k,0]
                wp[1,0]= W[i,k,1]
                wp[2,0]= W[i,k,2]
                W2= Average_AR1(I-i+1, K2, wp)
                XAvg2[0]=0
                XAvg2[1]=0
                XAvg2[2]= max(0,-L_max+ ( l_init + W[i, k, 1]))
                XAvg2[3]=0
                XAvg2[4]=0
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
    g.write("Time: %s seconds ----" %(time.time()- start_time))
    g.close()
    h.close()
    return lower_bound