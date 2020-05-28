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

def read_table(file_name):
    with open(file_name) as json_file:
        table = json.load(json_file)
    return table

class solveMDP:
    def __init__(self,
                 I,
                 K,
                 W,
                 l_init,
                 q_init,
                 c_init,
                 numProcess):
        self.K = K
        self.I = I
        self.config = read_parameters_form_json('Parameters.json')
        self.Price = np.multiply(np.multiply(W[:, :, 0], W[:, :, 1]), W[:, :, 2])
        self.Inflow = np.multiply(W[:, :, 3], W[:, :, 4])
        self.Condition = W[:, :, 5] 
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
        self.gran_theta1 = 2000
        self.gran_theta2 = 0.2
        largenumber = 2*L_max
        
        
        GenSum = 0
        UpSum = 0
        SpSum = 0
        RefSum = 0
        FlrSum = 0
        
        # loop over all sample paths
        for k in range(SubSamples):
            if self.Price[self.I, k] < 25:
                self.table = read_table('RobustTables/Table-modelrobust-category1-capacity36000.json')
            elif 25 <= self.Price[self.I, k] < 50:
                self.table = read_table('RobustTables/Table-modelrobust-category2-capacity36000.json')
            elif 50 <= self.Price[self.I, k] < 75:
                self.table = read_table('RobustTables/Table-modelrobust-category3-capacity36000.json')
            elif 75 <= self.Price[self.I, k] <= 100:
                self.table = read_table('RobustTables/Table-modelrobust-category4-capacity36000.json')
            else:
                self.table = read_table('RobustTables/Table-modelrobust-category5-capacity36000.json')
                

            # read the table of terminal value
            Month = '0'
            features = {}
            feature1 = []
            feature2 = []
            value = []
            for elem in self.table[Month]:
                features[(elem[0]*self.gran_theta1, elem[1]/100.)] = elem[2]
                feature1.append(elem[0]*self.gran_theta1)
                feature2.append(elem[1]/100.)
                value.append([elem[2]])
            n_feature1 = sorted(set(feature1))
            n_feature2 = sorted(set(feature2))

            Terminal_value = np.zeros((len(n_feature1), len(n_feature2)))
            for i, f1 in enumerate(n_feature1):
                for j, f2 in enumerate(n_feature2):
                    Terminal_value[i, j] = features[(f1, f2)]
            # Model
            perfectInfo = Model("GenAndUpg")
            # defining the variables
            
            
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
            for q in range(self.I+1):
                Cap.append(perfectInfo.addVar(lb=0 , vtype=GRB.CONTINUOUS,
                                              name="Cap[%d]" %q))    

            Res=[]
            for l in range(self.I+1):
                Res.append(perfectInfo.addVar(lb= L_min, ub=L_max, vtype=GRB.CONTINUOUS,
                                              name="Res[%d]" %l))  

            Con=[]
            for c in range(self.I +1):
                Con.append(perfectInfo.addVar(lb=0 , ub=1, vtype=GRB.CONTINUOUS,
                                              name="Con[%d]" %c))      

            Terminal_decision = perfectInfo.addVars(len(n_feature1), len(n_feature2), lb = 0, vtype = GRB.BINARY,\
                                                    name="Terminal_decision")
            theta1_bin=[]
            for item1 in range(len(n_feature1)):
                theta1_bin.append(perfectInfo.addVar(lb = 0, vtype = GRB.BINARY, 
                                                            name="theta1_bin[%d]" %(item1)))
            theta2_bin=[]
            for item2 in range(len(n_feature2)):
                theta2_bin.append(perfectInfo.addVar(lb = 0, vtype = GRB.BINARY, 
                                                            name="theta2_bin[%d]" %(item2)))
            

            perfectInfo.ModelSense = GRB.MAXIMIZE
            perfectInfo.setObjective(quicksum([quicksum([self.Price[time, k]* EnergyCoeff* \
                                         np.power(disc_fac, time)*Gen[time]-\
                                         C_U*np.power(disc_fac, time) *\
                                         Up[time] - C_R * np.power(disc_fac,time) * Ref[time] -\
                                         C_F* np.power(disc_fac,time)*Flr[time] for time in range(self.I)]),\
                                               np.power(disc_fac, self.I) * quicksum([Terminal_value[dim1, dim2]*Terminal_decision[dim1, dim2]\
                                               for dim1 in range(len(n_feature1)) for dim2 in range(len(n_feature2))\
                                                        ])\
                                              ]))
            
            # constraints
            perfectInfo.addConstr(
                    (quicksum(theta1_bin[item] for item in range(len(n_feature1))) == 1))
            perfectInfo.addConstr(
                    (quicksum(theta2_bin[item] for item in range(len(n_feature2))) == 1))

            for item1 in range(len(n_feature1)):
                for item2 in range(len(n_feature2)):
                    perfectInfo.addConstr(-theta1_bin[item1] - theta2_bin[item2] +\
                                              3 * Terminal_decision[item1, item2] <= 0)
                
            for item in range(len(n_feature1)):
                perfectInfo.addConstr(-Res[self.I] + (2*item -1) * self.gran_theta1/2. \
                                      - largenumber *(1-theta1_bin[item]) <= 0)
                perfectInfo.addConstr(Res[self.I] - (2*item +1) * self.gran_theta1/2. \
                                      - largenumber *(1-theta1_bin[item]) <= 0)
                
            for item in range(len(n_feature2)):
                perfectInfo.addConstr(-Con[self.I] + (2*(item) -1) * self.gran_theta2/2. \
                                      - largenumber *(1-theta2_bin[item]) <= 0)
                perfectInfo.addConstr(Con[self.I] - (2* item +1) * self.gran_theta2/2. \
                                      - largenumber *(1-theta2_bin[item]) <= 0)      
                  
          
                
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
                        (Gen[i]- Cap[i] <= 0), "Generation_cap1")

            for i in range(self.I):   
                perfectInfo.addConstr(
                        (Gen[i] - Res[i] <= 0), "Generation_cap2") 
                
#                perfectInfo.addConstr(
#                        (Gen[i]- Res[i]- self.Inflow[i, k] <= 0), "Generation_cap2")   
            
            for i in range(self.I):     
                perfectInfo.addConstr(
                        (Up[i]- 0.5* self.q_init * Ref[i] <= 0), "Upgrade_cap.")    


            for i in range(self.I):     
                perfectInfo.addConstr(
                        (Gen_b[i] + Ref[i] <= 1)) 
                
            for i in range(self.I):     
                perfectInfo.addConstr(
                        (Con[i] - 0.999 - M* Flr[i] <= 0)) 

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
            for i in range(self.I):
                perfectInfo.addConstr(
                        (Res[i+1]- Res[i]+ Gen[i] - self.Inflow[i , k] + Sp[i]==0))

            for i in range(self.I):
                perfectInfo.addConstr(
                        (Cap[i+1]- Cap[i] - Up[i] ==0 ))

            for i in range(self.I):
                perfectInfo.addConstr(
                        (Con[i+1]- Con[i] - (self.Condition[i, k])* Gen_b[i] + (Con[i]) * Ref[i] == 0))
                


            perfectInfo.write('GenAndUpg.lp') 

            
            perfectInfo.optimize()  
            print('l_init, c_init, q_init', self.l_init, self.c_init, self.q_init)
#            print('price, inflow, condition', self.Price[:,k], self.Inflow[:, k], 
#                 self.Condition[:, k])
            print('Gen[0]: {},Up[0]: {}, Sp[0]:{}, Ref[0]:{}, Flr[0]:{} \n'. format(
                             Gen[0].x, Up[0].x, Sp[0].x, Ref[0].x, Flr[0].x))


            if perfectInfo.status == GRB.Status.OPTIMAL:
                print('Optimal objective: %g' % perfectInfo.objVal)
                ub[k]= perfectInfo.objVal
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
            GenSum+= Gen[0].x

            UpSum+= Up[0].x
            SpSum+= Sp[0].x
            RefSum+= Ref[0].x 
            FlrSum+= Flr[0].x
        UB = np.sum(ub)
        return [GenSum, UpSum, SpSum, RefSum, FlrSum, UB]



       

     
       

        

