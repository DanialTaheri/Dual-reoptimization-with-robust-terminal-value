import numpy as np
import matplotlib.pyplot as plt
from ScenarioGeneratio import generate_Sample_Avr
from gurobipy import Model, Var, GRB, quicksum
import multiprocessing
from abc import ABCMeta, abstractmethod
import json
#import ScenarioGeneratio 



def SumList(List):
    finalSum = [0, 0, 0, 0, 0, 0] 
    #print("List SumList",List)
    for i, value in enumerate(List):
        #print("finalSum SumList",finalSum)
        if type(value) == list:
            for j, elem in enumerate(value):
                finalSum[j] = finalSum[j]+ elem
        else:
            finalSum[i] = finalSum[i] + value
    return finalSum 

def read_parameters_form_json(file_name):
    with open(file_name) as f:
        config = json.load(f)
    return config


class solveMDP:
    def __init__(self,
                 I,
                 K,
                 W,
                 l_init,
                 q_init,
                 c_init,
                 numProcess):
        self.config = read_parameters_form_json('Parameters.json')
        self.Price = np.multiply(np.multiply(W[:, :, 0], W[:, :, 1]), W[:, :, 2])
        self.Inflow = np.multiply(W[:, :, 3], W[:, :, 4])
        self.Condition = W[:, :, 5] 
        self.I = I
        self.K = K
        self.l_init = l_init
        self.q_init = q_init
        self.c_init = c_init
        print("Number of cpu:", multiprocessing.cpu_count()) 
        pool = multiprocessing.Pool(processes = numProcess)
        SubK = int(self.K/ numProcess) 
        results = pool.map(self.doParallelWork, [SubK for _ in range(numProcess)])
        #print(SumList(results))
        pool.close()
        pool.join()
        finalSum = SumList(results)
        # average of actions in the sample paths    
        GenAvg = finalSum[0]/self.K
        UpAvg = finalSum[1]/self.K
        SpAvg = finalSum[2]/self.K
        RefAvg = finalSum[3]/self.K
        FlrAvg = finalSum[4]/self.K
        UBAvg = finalSum[5]/self.K

        if RefAvg> 0.5:
            RefAvg=1
            UpAvg= UpAvg
            GenAvg= 0
        else:
            RefAvg=0
            UpAvg=0

        self.XAvg=[GenAvg, UpAvg, SpAvg, RefAvg, FlrAvg]
        self.UB = UBAvg
        
    def output(self):
        return self.UB, self.XAvg
                       
    def doParallelWork(self, SubK):
        Variables = self.solveMDP_Parallel(SubK)
        finalSum = SumList(Variables)
        return finalSum
    


    def solveMDP_Parallel(self, SubSamples):
        L_min =self.config['minLevel']
        L_max = self.config['maxLevel']
        ub = np.zeros(SubSamples)
        M = 2 * self.q_init  
        T_R_down = self.config['ref_down']
        T_F_down = self.config['failure_down']
        C_F = self.config['CostRf']
        C_R = self.config['CostRf']
        C_U = self.config['CostUp']
        disc_fac = self.config['disc_rate']
        EnergyCoeff = self.config['EnergyCoeff']
        self.gran = 200
        
        
        
        GenSum = 0
        UpSum = 0
        SpSum = 0
        RefSum = 0
        FlrSum = 0
        
    # loop over all sample paths
        for k in range(SubSamples):
            # Model
            perfectInfo = Model("GenAndUpg")
            ### defining the variables
            Gen=[]
            for x_G in range(self.I):
                Gen.append(perfectInfo.addVar(lb = 0, ub = 2*self.q_init, vtype = GRB.CONTINUOUS, 
                                              name="Gen[%d]" %x_G))
            
            Up=[]    
            for x_U in range(self.I):
                    Up.append(perfectInfo.addVar(lb=0, ub = 0.5* self.q_init, vtype=GRB.CONTINUOUS,  name="Up[%d]" %x_U))

            Sp=[]    
            for x_S in range(self.I):
                Sp.append(perfectInfo.addVar(lb=0, vtype=GRB.CONTINUOUS,name="Sp[%d]" %x_S))    

            Ref=[]    
            for x_R in range(self.I):
                Ref.append(perfectInfo.addVar(lb=0, vtype=GRB.BINARY,
                                              name="Ref[%d]" %x_R))     
            # generation binary var
            Gen_b=[]
            for xi_G in range(self.I):
                Gen_b.append(perfectInfo.addVar(lb=0, vtype=GRB.BINARY,
                                                name="Gen_b[%d]" %xi_G))   

            Flr=[]
            for xi_F in range(self.I):
                Flr.append(perfectInfo.addVar(lb=0, vtype=GRB.BINARY,
                                              name="Flr[%d]" %xi_F))      
            
            Cap=[]
            for q in range(self.I):
                Cap.append(perfectInfo.addVar(lb=0 , vtype=GRB.CONTINUOUS,
                                              name="Cap[%d]" %q))    

            Res=[]
            for l in range(self.I):
                Res.append(perfectInfo.addVar(lb= L_min, ub=L_max, vtype=GRB.CONTINUOUS,
                                              name="Res[%d]" %l))  

            Con=[]
            for c in range(self.I):
                Con.append(perfectInfo.addVar(lb=0 , ub=1, vtype=GRB.CONTINUOUS,
                                              name="Con[%d]" %c))      

            perfectInfo.ModelSense = GRB.MAXIMIZE
            perfectInfo.setObjective(quicksum([self.Price[time, k]* EnergyCoeff*np.power(disc_fac, time)*Gen[time]-\
                                         C_U*np.power(disc_fac, time) * Up[time] - C_R * np.power(disc_fac, time) * Ref[time] -\
                                         C_F* np.power(disc_fac,time)*Flr[time] for time in range(self.I)]))

            # constraints
            # Initial constraints
            ####################################################
            perfectInfo.addConstr(
                    (Res[0]- self.l_init == 0), "initial_reservior")

            perfectInfo.addConstr(
                    (Cap[0]- self.q_init == 0), "initial_capacity")

            perfectInfo.addConstr(
                    (Con[0]- self.c_init == 0), "initial_condition")    
            ########################################################
            
            
            for i in range(self.I):  
                perfectInfo.addConstr(
                        (Gen[i]- Cap[i] <= 0))

            for i in range(self.I):     
                perfectInfo.addConstr(
                        (Gen[i]- Res[i] <= 0))   

            for i in range(self.I):     
                perfectInfo.addConstr(
                        (Up[i]- 0.5* self.q_init * Ref[i] <= 0))    

            for i in range(self.I):     
                perfectInfo.addConstr(
                        (Gen_b[i] + Ref[i] <= 1)) 
                
            for i in range(self.I):     
                perfectInfo.addConstr(
                        (Con[i] - 0.99 - M* Flr[i] <= 0)) 

            for i in range(self.I):     
                perfectInfo.addConstr(
                        ( Gen[i] - M* Gen_b[i] <= 0)) 

            for i in range(self.I):     
                perfectInfo.addConstr(
                        (quicksum(Gen[i+j] for j in range(min(T_R_down, self.I-i )))- T_R_down* M * (1-Ref[i]) <=0))

            for i in range(self.I):     
                perfectInfo.addConstr(
                        (quicksum(Gen[i+j] for j in range(min(T_F_down, self.I-i )))- T_F_down* M* (1-Flr[i]) <=0))
                        
                
            # Transition function
            #########################################################
            # generation happens after receiving inflow
            for i in range(self.I-1):
                perfectInfo.addConstr(
                        (Res[i+1]- Res[i]+ Gen[i] - self.Inflow[i , k] + Sp [i]==0))

            for i in range(self.I-1):
                perfectInfo.addConstr(
                        (Cap[i+1]- Cap[i] - Up[i] ==0 ))

            for i in range(self.I-1):
                perfectInfo.addConstr(
                        (Con[i+1]- Con[i] - (self.Condition[i, k])* Gen_b[i] + (Con[i]) * Ref[i] == 0))
                

            perfectInfo.write('GenAndUpg.lp') 
            print('l_init, c_init, q_init', self.l_init, self.c_init, self.q_init)
            print('price, inflow, condition', self.Price[:,k], self.Inflow[:, k], 
                 self.Condition[:, k])

            perfectInfo.optimize()  
            print('Gen[0]: {},Up[0]: {}, Sp[0]:{}, Ref[0]:{}, Flr[0]:{} \n'. format(
                             Gen[0].x, Up[0].x, Sp[0].x, Ref[0].x, Flr[0].x))


            print('Sample path: {:.4f}'.format(k))
            for i in range(self.I):
                print('Week {}: Optimal generation: {:.4f}'. format(i, Gen[i].x))
            if perfectInfo.status == GRB.Status.OPTIMAL:
                print('Optimal objective: %g' % perfectInfo.objVal)
                ub[k]= perfectInfo.objVal

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
            GenSum+= Gen[0].x

            UpSum+= Up[0].x
            SpSum+= Sp[0].x
            RefSum+= Ref[0].x 
            FlrSum+= Flr[0].x
        UB = np.sum(ub)
        return [GenSum, UpSum, SpSum, RefSum, FlrSum, UB]





       

     
       

        

