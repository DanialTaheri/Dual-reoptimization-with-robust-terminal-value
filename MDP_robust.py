# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:52:40 2019

@author: smohse3
"""

import numpy as np
import matplotlib.pyplot as plt
import inputProcessing
import ScenarioGeneratio
from gurobipy import Model, Var, GRB, quicksum
import pdb
#import ScenarioGeneratio 

def solveMDP_robust(gp, I, K, W, l_init, q_init, c_init, I2):
    GenAvg=0.0
    UpAvg=0.0
    SpAvg=0.0
    RefAvg=0.0
    FlrAvg=0.0
    L_min= 50
    L_max=1200
    ub=np.zeros(K)
    M=1000000   
    T_R_down=2
    T_F_down=4
    C_F = 100
    Gamma_tot_w = 1
    Gamma_tot_si = 1
    Gamma_w = 2
    Gamma_si = 3
    pi_w = 0.2
    pi_si = 0.2
    rho = 0.05
    nu = 0.01
    # loop over all sample paths
    for k in range(K):
        # Model
        perfectInfo = Model("GenAndUpg")
        perfectInfo.Params.nonconvex = 2
        Gen_mdp=[]
        for x_G in range(I):
            Gen_mdp.append(perfectInfo.addVar(lb=0 , vtype=GRB.CONTINUOUS, 
                                          obj=W[x_G, k, 0]*gp[5]*np.power(gp[6], x_G),
                                          name="Gen_mdp[%d]" %x_G))
                       
        Up_mdp=[]    
        for x_U in range(I):
            Up_mdp.append(perfectInfo.addVar(lb=0, vtype=GRB.CONTINUOUS, 
                                             obj= - gp[7]*np.power(gp[6], x_U),
                                             name="Up_mdp[%d]" %x_U))
    
        Sp_mdp=[]    
        for x_S in range(I):
            Sp_mdp.append(perfectInfo.addVar(lb=0, vtype=GRB.CONTINUOUS, 
                                         obj=0,
                                         name="Sp_mdp[%d]" %x_S))    

        Ref_mdp=[]    
        for x_R in range(I):
            Ref_mdp.append(perfectInfo.addVar(lb=0, vtype=GRB.BINARY, 
                                          obj= - gp[8]*np.power(gp[6],x_R),
                                          name="Ref_mdp[%d]" %x_R))     
    
        Gen_b_mdp = []
        for xi_G in range(I):
            Gen_b_mdp.append(perfectInfo.addVar(lb=0, vtype=GRB.BINARY, 
                                            obj=0,
                                            name="Gen_b_mdp[%d]" %xi_G))   

        Flr_mdp = []
        for xi_F in range(I):
            Flr_mdp.append(perfectInfo.addVar(lb=0, vtype=GRB.BINARY, 
                                          obj=- C_F* np.power(gp[6], xi_F),
                                          name="Flr_mdp[%d]" %xi_F))      

        Cap_mdp=[]
        for q in range(I):
            Cap_mdp.append(perfectInfo.addVar(lb=0 , vtype=GRB.CONTINUOUS, 
                                          obj=0,
                                          name="Cap_mdp[%d]" %q))    
        
        Res_mdp=[]
        for l in range(I):
            Res_mdp.append(perfectInfo.addVar(lb= L_min, ub=L_max, vtype=GRB.CONTINUOUS, 
                                          obj=0,
                                          name="Res_mdp[%d]" %l))  
        
        Con_mdp=[]
        for c in range(I):
            Con_mdp.append(perfectInfo.addVar(lb=0 , ub=1, vtype=GRB.CONTINUOUS, 
                                          obj=0,
                                          name="Con_mdp[%d]" %c))      
        #########################################################################
        #robust part
        Gen_rb=[]
        for y_G in range(I2):
            Gen_rb.append(perfectInfo.addVar(lb=0 , vtype=GRB.CONTINUOUS, 
                                          name="Gen_rb[%d]" %y_G))
        #y_F               
        Flr_rb=[]    
        for y_F in range(I2):
            Flr_rb.append(perfectInfo.addVar(lb=0, vtype=GRB.BINARY, 
                                             name="Flr_rb[%d]" %y_F))
        #y_L
        Res_rb=[]    
        for y_L in range(I2):
            Res_rb.append(perfectInfo.addVar(lb=L_min, ub = L_max, vtype=GRB.CONTINUOUS, 
                                         name="Res_rb[%d]" %y_L))    
        #deterioration rate
        det_rb=[]    
        for gamma in range(I2):
            det_rb.append(perfectInfo.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, 
                                          name="det_rb[%d]" %gamma))  
        
        #dual variable
        dual_rb=[]
        for alpha in range(6*(I2-1)+8):
            if alpha == 1 or alpha==2 or alpha >= 4*(I2-1)+6:
                dual_rb.append(perfectInfo.addVar(lb= -10000, ub = 100000 , vtype=GRB.CONTINUOUS, 
                                          name="dual_rb[%d]" %alpha))     
            else:
                dual_rb.append(perfectInfo.addVar(lb = 0 , ub = 100000, vtype=GRB.CONTINUOUS, 
                              name="dual_rb[%d]" %alpha))  

        
        
        Coeff = coef_func(Gamma_w, Gamma_si, C_F, gp[6], I2, Gen_rb, Res_rb, Flr_rb, Gamma_tot_w, Gamma_tot_si, W[-1, k, 0], W[-1, k, 0])
        perfectInfo.setObjective(quicksum([quicksum([W[time, k, 0]*gp[5]*np.power(gp[6], time)*Gen_mdp[time]- gp[7]*np.power(gp[6], time) * Up_mdp[time] - gp[8]*np.power(gp[6],time)* Ref_mdp[time] - C_F* np.power(gp[6], time)*Flr_mdp[time] for time in range(I)]),
                                 quicksum([quicksum(dual_rb[itr] * Coeff.output()[itr,0] for itr in range(3*(I2-1)+5)),
                                 quicksum([dual_rb[3*(I2-1)+5]* (Res_rb[0]- Cap_mdp[-1] + Gen_mdp[-1]), quicksum(dual_rb[3*(I2-1)+6+itr] * (Res_rb[itr+1] - Res_rb[itr] + Gen_rb[itr]) for itr in range(I2-1))])]) + quicksum(0* dual_rb[itr + 4*(I2-1)+6] for itr in range(2*I2))]), GRB.MAXIMIZE)
        
            
        
    
###############################################################################
        #MDP constraints
        perfectInfo.addConstr(
                (Res_mdp[0]- l_init == 0), "initial reservior")
        
     
        perfectInfo.addConstr(
                (Cap_mdp[0]- q_init == 0), "initial capacity")
        
 
        perfectInfo.addConstr(
                (Con_mdp[0]- c_init == 0), "initial condition")    
    
        for i in range(I):  
            perfectInfo.addConstr(
                    (Gen_mdp[i]- Cap_mdp[i]  <= 0), "Generation")


        for i in range(I):     
            perfectInfo.addConstr(
                    (Gen_mdp[i]- Res_mdp[i] - W[i , k, 1]  <= 0))   

        for i in range(I):     
            perfectInfo.addConstr(
                    (Up_mdp[i]- gp[4] * Ref_mdp[i] <= 0))    
    

        for i in range(I):     
            perfectInfo.addConstr(
                    ( Gen_b_mdp[i] + Ref_mdp[i] <= 1)) 
            
        for i in range(I):     
            perfectInfo.addConstr(
                    ( Gen_mdp[i] - M* Gen_b_mdp[i] <= 0)) 

        for i in range(I):     
            perfectInfo.addConstr(
                    (Ref_mdp[i]- Flr_mdp[i] <= 0)) 

        for i in range(I):     
            perfectInfo.addConstr(
                    (quicksum(Gen_mdp[i+j] for j in range(min(T_R_down, I-i )))- M* (1-Ref_mdp[i]) <=0))
                               
        for i in range(I):     
            perfectInfo.addConstr(
                    (quicksum(Gen_mdp[i+j] for j in range(min(T_F_down, I-i )))- M* (1-Flr_mdp[i]) <=0))
    
        for i in range(I-1):
            perfectInfo.addConstr(
                    (Res_mdp[i+1]- Res_mdp[i]+ Gen_mdp[i] - W[i , k, 1] + Sp_mdp[i]==0 ))
        
        for i in range(I-1):
            perfectInfo.addConstr(
                    (Cap_mdp[i+1]- Cap_mdp[i] - Up_mdp[i] ==0 ))
          
        for i in range(I-1):
            perfectInfo.addConstr(
                    (Con_mdp[i+1]- Con_mdp[i] -  W[i, k, 2]* Gen_b_mdp[i] -(0- Con_mdp[i])* Ref_mdp[i] ==0 ))
            
        ################################################################
        # Robust constraints
        
        
        perfectInfo.addConstr(
        (det_rb[0]- Con_mdp[-1] == 0), "robust initial condition") 
        
        

        for itr in range(I2):  
#            perfectInfo.addConstr(
#                    (Gen_rb[itr]- Res_rb[itr] <= 0), "Generation and reservior level")
            
            perfectInfo.addConstr(
                (Gen_rb[itr]- Cap_mdp[-1] <= 0), "Generation and capacity")            
        for itr in range(I2-1):     
            perfectInfo.addConstr(
                    ( Flr_rb[itr] - det_rb[itr] -nu + det_rb[itr+1] >= 0))
            
        C = np.zeros((4*I2+1, 1))

        C[-1] = 1
        
        Coef_Const = coef_Const(pi_w, pi_si, rho, gp[6], Gen_rb, I2)
        for i in range(4*I2+1):
            perfectInfo.addConstr(
                    (quicksum(np.transpose(Coef_Const.output())[itr,j]*dual_rb[j] for j in range(6*(I2-1)+8)) - C[itr] <= 0))
        
        perfectInfo.write('GenAndUpg.lp') 
     
            
        perfectInfo.optimize()  
        print('Up[0]: {}, Sp[0]:{}, Ref[0]:{}, Flr[0]:{} \n'. format(
                         Up_mdp[0].x, Sp_mdp[0].x, Ref_mdp[0].x, Flr_mdp[0].x ))
        # pdb.set_trace() 
#        print('Power: {}, Inflow: {}, Detereoration: {}\n'. format(
#                W[i, 0, 0], W[i, 0, 1],  W[i, 0, 2]))  
        
        print('Sample path: {:.4f}'.format(
                k ))
        if perfectInfo.status == GRB.Status.OPTIMAL:
            print('Optimal objective: %g' % perfectInfo.objVal)
            ub[k]= -perfectInfo.objVal
#        else:
#            ub[k]= M
        elif perfectInfo.status == GRB.Status.INF_OR_UNBD:
            print('Model is infeasible or unbounded')
            exit(0)
        elif perfectInfo.status == GRB.Status.INFEASIBLE:
            print('Model is infeasible')
            exit(0)
        elif perfectInfo.status == GRB.Status.UNBOUNDED:
            print('Model is unbounded')
            exit(0)
        else:  
            print('Optimization ended with status %d' % perfectInfo.status)
        
        #refering to decisions at current period
       # pdb.set_trace()    
        GenAvg+= Gen_mdp[0].x
        print('Optimal generation: {:.4f}'. format( Gen_mdp[0].x))
        UpAvg+= Up_mdp[0].x
        SpAvg+= Sp_mdp[0].x
        RefAvg+= Ref_mdp[0].x 
        FlrAvg+= Flr_mdp[0].x
        
    # average of actions in the sample paths    
    GenAvg=GenAvg/K
    UpAvg=UpAvg/K
    SpAvg=SpAvg/K
    RefAvg= RefAvg/K
    FlrAvg=FlrAvg/K
    
    if RefAvg> 0.5:
        RefAvg=1
        UpAvg= UpAvg
        GenAvg= 0
    else:
        RefAvg=0
        UpAvg=0

       
    XAvg=[GenAvg, UpAvg, SpAvg, RefAvg, FlrAvg]
    
    return ub, XAvg  



class coef_func:
    def __init__(self, a, b, c, d, e , f, g, h, i, j, k, l):
        self.Gamma_w = a
        self.Gamma_si = b
        self.C_F = c
        self.delta = d
        self.I = e
        self.Gen = f
        self.Cap = g
        self.Flr = h
        self. Gamma_tot_w = i
        self.Gamma_tot_si = j
        self.inflow_init = k
        self.Price_init = l
    def flr_cost(self):
        ans = 0
        for i in range(self.I):
            ans -= self.delta**(i-1)*self.C_F* self.Flr[i]
        return ans
    
    def Budgt_init(self):
        return np.asarray([[self.inflow_init], [self.Price_init]]).reshape(-1,1)
    def Budgt_w1(self):
        return np.asarray([-self.Gamma_w] * (self.I-1)).reshape(-1,1)
    def Budgt_w2(self):
        return np.asarray([-self.Gamma_w] * (self.I-1)).reshape(-1,1)
    def Budgt_si(self):
        return np.asarray([-self.Gamma_si] * (self.I-1)).reshape(-1,1)
    
    def total_Budgt(self):
        return np.asarray([[-self.Gamma_tot_w], [-self.Gamma_tot_si]]).reshape(-1,1)    
    def output(self):
        return np.vstack((self.flr_cost(), self.Budgt_init(), self.Budgt_w1(), self.Budgt_w2(), self.Budgt_si(), self.total_Budgt()))
    
    
class coef_Const:
    def __init__(self, a, b, c, d, e, f):
        self.pi_w = a
        self.pi_si = b
        self.rho = c
        self.delta = d
        self.Gen = e
        self.I = f
    def first_Const(self):
        X1 = [-self.Gen[i- 3* self.I] * self.delta**(i - 3*self.I) for i in range(3*self.I, 4*self.I)]
        X2 = [0 for i in range(3*self.I)]
        return np.vstack((np.asarray(X2).reshape(-1,1), np.asarray(X1).reshape(-1,1), [1]))
    
    def Second_Const(self):
        temp = np.zeros((1, 4*self.I+1))
        temp[0, 0] = 1
        return temp
    
    def Third_Const(self):
        temp = np.zeros((1, 4*self.I+1))
        temp[0, self.I] = 1
        return temp
    
    def Fourth_Const(self):
        temp = np.zeros((self.I-1, 4*self.I+1))
        for i in range(self.I-1):
            temp[i,i] = self.pi_w
            temp[i, i+1] = -1
            
        return temp
            
    def Fifth_Const(self):
        temp = np.zeros((self.I-1, 4*self.I+1))
        for i in range(self.I-1):
            temp[i,i] = -self.pi_w
            temp[i, i+1] = 1
        return temp
            
    def Sixth_Const(self):
        temp = np.zeros((self.I-1, 4*self.I+1))
        for i in range(self.I-1):
            temp[i,i] = self.pi_w*self.rho
            temp[i, i+1] = -self.rho
            temp[i, i+self.I] = -self.pi_si
            temp[i, i+self.I+1] = 1
        return temp
            
    def Seventh_Const(self):
        temp = np.zeros((2, 4*self.I+1))
        temp[0, 0] = self.pi_w
        temp[1, 0] = -self.pi_w * self.rho
        temp[0, 1:self.I-1]= -1 + self.pi_w
        temp[1, 1:self.I-1]= (1- self.pi_w)*self.rho
        temp[0, self.I-1] = -1
        temp[1, self.I-1] = self.rho
        temp[ 1, self.I ] =  self.pi_si
        temp[ 1, self.I+1: 2*self.I-1]= -1+ self.pi_si
        temp[1, 2*self.I-1] = -1
        return temp
        
    def Eight_Const(self):
        temp = np.zeros((self.I, 4*self.I+1))
        temp[: , 2*self.I:3*self.I] = np.eye(self.I)
        return temp
    
    def Ninth_Const(self):
        temp = np.zeros((self.I, 4*self.I+1))
        for i in range(self.I):
            temp[i,i] = 1
            temp[i, i+ 2*self.I] = -1  
        return temp 
    
    def Tenth_Const(self):
        temp = np.zeros((self.I, 4*self.I+1))
        for i in range(self.I):
            temp[i,i+ self.I] = 1
            temp[i, i+ 3*self.I] = -1 
        return temp 
    
    
    def output(self):
        return np.vstack((np.transpose(self.first_Const()), self.Second_Const(), self.Third_Const(),  self.Fourth_Const(), self.Fifth_Const(), self.Sixth_Const(),\
                         self.Seventh_Const(), self.Eight_Const(), self.Ninth_Const(), self.Tenth_Const()))      

        


       

        

