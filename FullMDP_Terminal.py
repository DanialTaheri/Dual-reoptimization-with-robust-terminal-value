import numpy as np
#import matplotlib.pyplot as plt
from ScenarioGeneratio import generate_Sample_Avr
from gurobipy import Model, Var, GRB, quicksum
import multiprocessing
from abc import ABCMeta, abstractmethod
import json
import collections
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
        self.W = W
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
        Upper_bound = np.zeros(SubSamples)
        M = 2 * self.q_init  
        T_R_down = self.config['ref_down']
        T_F_down = self.config['failure_down']
        C_F = self.config['CostRf']
        C_R = self.config['CostRf']
        C_U = self.config['CostUp']
        disc_fac = self.config['disc_rate']
        EnergyCoeff = self.config['EnergyCoeff']
        model_type = self.config['model_type']


        
        GenSum = 0
        UpSum = 0
        SpSum = 0
        RefSum = 0
        FlrSum = 0
        
        # loop over all sample paths
        Table_nb = []
        for k in range(SubSamples):
            nb_bins = 2
            threshold = 0.63
            nb_features = 4
            # [res_coef, cond_coef, intercept]
            # Feature coefficients extracted from robust tables
            beta = np.zeros((nb_features, nb_bins))
            
            if model_type == 'robust':
                if self.W[self.I, k, 2] < 25:
                    beta[0, :] = [1.5543, 0.585]
                    beta[1, :] = [-2355202.905, -8124486.412]
                    beta[2, :] = [103.418, 44.1719]
                    beta[3, :] = [460480.896, 6746857.459]
                elif 25 <= self.W[self.I, k, 2] < 40:
                    beta[0, :] = [3.272, 1.355]
                    beta[1, :] = [-4197748.55, -14395691]
                    beta[2, :] = [181.702, 77.205]
                    beta[3, :] = [788321, 11931227]
                elif 40 <= self.W[self.I, k, 2] < 55:
                    beta[0, :] = [5.197, 2.1657]
                    beta[1, :] = [-6487780.282, -22166358.884]
                    beta[2, :] = [279.166, 117.645]
                    beta[3, :] = [1215310, 18383787]
                elif 55 <= self.W[self.I, k, 2] <= 70:
                    beta[0, :] = [7.816748016329668, 3.3523]
                    beta[1, :] = [-9348774.127, -31900687.36]
                    beta[2, :] = [400.979, 168.6723]
                    beta[3, :] = [1726348, 26437073]
                else:
                    beta[0, :] = [14.3146, 6.2848]
                    beta[1, :] =  [-16522199.781, -56264542.861]
                    beta[2, :] = [705.426, 295.418]
                    beta[3, :] = [705.426, 46618850]

                    
                    
            elif model_type == 'nominal':
                if self.W[self.I, k, 2] < 25:
                    beta[0, :] = [0.6339643048973409, 0.32416678020887013]
                    beta[1, :] = [-6922512.525771472, -23273109.217998378]
                    beta[2, :] = [14017378.79, 24210679.434]

                elif 25 <= self.W[self.I, k, 2] < 40:
                    beta[0, :] = [1.1141278155210295, 0.568316219837665]
                    beta[1, :] = [-12233805.510731239, -41187870.70060356]
                    beta[2, :] = [24798217.011, 42854854.42]
                    
                elif 40 <= self.W[self.I, k, 2] < 55:
                    beta[0, :] = [1.6757386471552294, 0.8533672154817823]
                    beta[1, :] = [-18495514.2059734, -62249939.595300876]
                    beta[2, :] = [37484193.9,  64767347.22]
                    
                elif 55 <= self.W[self.I, k, 2] <= 70:
                    beta[0, :] = [2.429242996467785, 1.2352535933265176]
                    beta[1, :] = [-26879703.91652888, -90470858.94148476]
                    beta[2, :] = [54479857.59, 94132023.05]
                    
                else:
                    beta[0, :] = [4.256425664491066, 2.1621733310195226]
                    beta[1, :] =  [-47227835.343741596, -158919486.96210086]
                    beta[2, :] = [95706049.9088, 165347979.473]
                    

            # Model
            perfectInfo = Model("GenAndUpg")
            perfectInfo.Params.TimeLimit= 50
            #if self.c_init > threshold:
            #perfectInfo.Params.nonconvex = 2
            # defining the variables
            
            Gen=[]
            for x_G in range(self.I):
                Gen.append(perfectInfo.addVar(lb = 0, ub = 2*self.q_init, vtype = GRB.CONTINUOUS, 
                                              name="Gen[%d]" %x_G))
            
            Up=[]    
            for x_U in range(self.I):
                    Up.append(perfectInfo.addVar(lb=0, ub = 0.5 * self.q_init, vtype=GRB.CONTINUOUS,  name="Up[%d]" %x_U))

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
                
            theta = []
            for item in range(nb_bins):
                theta.append(perfectInfo.addVar(lb = 0, vtype = GRB.BINARY, 
                                                            name="theta[%d]" %(item)))
                
            
            Coef_terminal = perfectInfo.addVars(range(nb_features), range(nb_bins), vtype = GRB.CONTINUOUS)
            
#             Coef_terminal = []
#             for item in range(nb_features):
#                 Coef_terminal.append(perfectInfo.addVar(lb = -10e+9, vtype = GRB.CONTINUOUS, 
#                                                             name="Coef_terminal[%d]" %(item)))
                

            perfectInfo.ModelSense = GRB.MAXIMIZE
            perfectInfo.setObjective(quicksum([quicksum([self.Price[time, k]* EnergyCoeff* \
                                        np.power(disc_fac, time)*Gen[time]-\
                                        C_U*np.power(disc_fac, time) *Up[time] -\
                                        C_R * np.power(disc_fac,time) * Ref[time] -\
                                        C_F* np.power(disc_fac,time)*Flr[time] for time in range(self.I)]),\
                                      np.power(disc_fac,self.I)*quicksum([Coef_terminal[_ft,_bin]*\
                                      beta[_ft, _bin] for _ft in range(nb_features) for _bin in range(nb_bins)])
                                              ]))
    
            perfectInfo.addConstr(
                    (quicksum(theta[item] for item in range(nb_bins)) == 1))
            
#  
                
            
#             for feature in range(nb_features):
#                 perfectInfo.addConstr(
#                     (Coef_terminal[feature] - beta[feature, 0] * theta[0] - beta[feature, 1] * theta[1]  == 0)) 
            U = [L_max, 1, 2* self.q_init]
            L = [0, 0, self.q_init]
            Var = [Res[self.I], Con[self.I], Cap[self.I]]
            for _ft in range(nb_features-1):
                for _bin in range(nb_bins):
                    perfectInfo.addConstr(
                        (Coef_terminal[_ft, _bin] - U[_ft] * theta[_bin] <= 0)) 
                    perfectInfo.addConstr(
                        (Coef_terminal[_ft, _bin] - Var[_ft] + L[_ft]*(1- theta[_bin]) <= 0)) 
                    perfectInfo.addConstr(
                        (-Coef_terminal[_ft, _bin] + Var[_ft] - U[_ft]*(1- theta[_bin]) <= 0))
            for _bin in range(nb_bins):
                perfectInfo.addConstr(
                    (Coef_terminal[3, _bin] - theta[_bin] == 0)) 
                
            perfectInfo.addConstr(
                     (Con[self.I] - threshold - (1 - theta[0]) <= 0))

            perfectInfo.addConstr(
                    (-Con[self.I] + threshold - (1 - theta[1]) <= 0))

                
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
                        (Gen[i]- Cap[i] <= 0), "Generation_cap1_%d"%i)

            for i in range(self.I):   
                perfectInfo.addConstr(
                        (Gen[i] - Res[i] <= 0), "Generation_cap2_%d"%i) 
                

            for i in range(self.I):     
                perfectInfo.addConstr(
                        (Up[i]- 0.5 * self.q_init * Ref[i] <= 0), "Upgrade_cap_%d"%i)    

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
                
            # fixing the refurbishment 8 weeks before the end of MDP to zero
            for i in range(max(self.I-1- T_R_down, 0), self.I):
                perfectInfo.addConstr(
                        (Ref[i] == 0))

            perfectInfo.write('GenAndUpg.lp') 

            perfectInfo.optimize()  
            #print("theta_0: {}, theta_1:{}".format(theta[0].x, theta[1].x))


            if perfectInfo.status == GRB.Status.OPTIMAL:
                print('Optimal objective: %g' % perfectInfo.objVal)
                Upper_bound[k]= perfectInfo.objVal
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
            
#             for _ft in range(nb_features):
#                 for _bin in range(nb_bins):
#                     print("Coefficient[{}, {}]: {} ".format(_ft, _bin, Coef_terminal[_ft, _bin].x))
            
#             for _bin in range(nb_bins):
#                 print("theta[{}]: {} ".format(_bin, theta[_bin].x))
                
#             print(quicksum([Coef_terminal[_ft,_bin].x*\
#                                       beta[_ft, _bin] for _ft in range(nb_features) for _bin in\
#                                                       range(nb_bins)]))
            #refering to decisions at current period    
            GenSum+= Gen[0].x

            UpSum+= Up[0].x
            SpSum+= Sp[0].x
            RefSum+= Ref[0].x 
            FlrSum+= Flr[0].x
#             path = "config_bounds.json"
#             with open(path, 'a') as f:
#                 f.write(str(perfectInfo.MIPGap))
#                 f.write('\n')
        
        UB = np.sum(Upper_bound)
        return [GenSum, UpSum, SpSum, RefSum, FlrSum, UB]