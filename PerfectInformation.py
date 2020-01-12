import numpy as np
import matplotlib.pyplot as plt
import inputProcessing
import ScenarioGeneratio
from gurobipy import Model, Var, GRB, quicksum
import pdb
#import ScenarioGeneratio 





def solvePerfectInformation(gp, I, K, W, l_init, q_init, c_init):
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
# loop over all sample paths
    for k in range(K):
# Model
        perfectInfo = Model("GenAndUpg")

        Gen=[]
        for x_G in range(I):
            Gen.append(perfectInfo.addVar(lb=0 , vtype=GRB.CONTINUOUS, 
                                          obj=-W[x_G, k, 0]*gp[5]*np.power(gp[6], x_G),
                                          name="Gen[%d]" %x_G))
                       
        Up=[]    
        for x_U in range(I):
                Up.append(perfectInfo.addVar(lb=0, vtype=GRB.CONTINUOUS, 
                                             obj=gp[7]*np.power(gp[6], x_U),
                                             name="Up[%d]" %x_U))
    
        Sp=[]    
        for x_S in range(I):
            Sp.append(perfectInfo.addVar(lb=0, vtype=GRB.CONTINUOUS, 
                                         obj=0,
                                         name="Sp[%d]" %x_S))    

        Ref=[]    
        for x_R in range(I):
            Ref.append(perfectInfo.addVar(lb=0, vtype=GRB.BINARY, 
                                          obj=gp[8]*np.power(gp[6],x_R),
                                          name="Ref[%d]" %x_R))     
    
        Gen_b=[]
        for xi_G in range(I):
            Gen_b.append(perfectInfo.addVar(lb=0, vtype=GRB.BINARY, 
                                            obj=0,
                                            name="Gen_b[%d]" %xi_G))   

        Flr=[]
        for xi_F in range(I):
            Flr.append(perfectInfo.addVar(lb=0, vtype=GRB.BINARY, 
                                          obj=(100)* np.power(gp[6], xi_F),
                                          name="Flr[%d]" %xi_F))      

        Cap=[]
        for q in range(I):
            Cap.append(perfectInfo.addVar(lb=0 , vtype=GRB.CONTINUOUS, 
                                          obj=0,
                                          name="Cap[%d]" %q))    
        
        Res=[]
        for l in range(I):
            Res.append(perfectInfo.addVar(lb= L_min, ub=L_max, vtype=GRB.CONTINUOUS, 
                                          obj=0,
                                          name="Res[%d]" %l))  
        
        Con=[]
        for c in range(I):
            Con.append(perfectInfo.addVar(lb=0 , ub=1, vtype=GRB.CONTINUOUS, 
                                          obj=0,
                                          name="Con[%d]" %c))      
        #########################################################################
        #robust part
        
    
    
        perfectInfo.modelSense = GRB.MINIMIZE
    
     
        perfectInfo.addConstr(
                (Res[0]- l_init == 0), "initial reservior")
        
     
        perfectInfo.addConstr(
                (Cap[0]- q_init == 0), "initial capacity")
        
 
        perfectInfo.addConstr(
                (Con[0]- c_init == 0), "initial condition")    
    
        for i in range(I):  
            perfectInfo.addConstr(
                    (Gen[i]- Cap[i] <= 0), "Generation")


        for i in range(I):     
            perfectInfo.addConstr(
                    (Gen[i]- Res[i] <= 0))   

        for i in range(I):     
            perfectInfo.addConstr(
                    (Up[i]- gp[4] * Ref[i] <= 0))    
    

        for i in range(I):     
            perfectInfo.addConstr(
                    ( Gen_b[i] + Ref[i] <= 1)) 
            
        for i in range(I):     
            perfectInfo.addConstr(
                    ( Gen[i] - M* Gen_b[i] <= 0)) 

        for i in range(I):     
            perfectInfo.addConstr(
                    (Ref[i]- Flr[i] <= 0)) 

        for i in range(I):     
            perfectInfo.addConstr(
                    (quicksum(Gen[i+j] for j in range(min(T_R_down, I-i )))- M* (1-Ref[i]) <=0))
                               
        for i in range(I):     
            perfectInfo.addConstr(
                    (quicksum(Gen[i+j] for j in range(min(T_F_down, I-i )))- M* (1-Flr[i]) <=0))
    
        for i in range(I-1):
            perfectInfo.addConstr(
                    (Res[i+1]- Res[i]+ Gen[i] - W[i , k, 1] + Sp [i]==0 ))
        
        for i in range(I-1):
            perfectInfo.addConstr(
                    (Cap[i+1]- Cap[i] - Up[i] ==0 ))
          
        for i in range(I-1):
            perfectInfo.addConstr(
                    (Con[i+1]- Con[i] -  W[i, k, 2]* Gen_b[i] -(0- Con[i])* Ref[i] ==0 ))
        
        perfectInfo.write('GenAndUpg.lp') 
     
            
        perfectInfo.optimize()  
        print('Up[0]: {}, Sp[0]:{}, Ref[0]:{}, Flr[0]:{} \n'. format(
                         Up[0].x, Sp[0].x, Ref[0].x, Flr[0].x ))
        print('Power: {}, Inflow: {}, Detereoration: {}\n'. format(
                W[i, 0, 0], W[i, 0, 1],  W[i, 0, 2] ))  
        
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
        GenAvg+= Gen[0].x
        print('Optimal generation: {:.4f}'. format( Gen[0].x))
        UpAvg+= Up[0].x
        SpAvg+= Sp[0].x
        RefAvg+= Ref[0].x 
        FlrAvg+= Flr[0].x
        
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




       

        

