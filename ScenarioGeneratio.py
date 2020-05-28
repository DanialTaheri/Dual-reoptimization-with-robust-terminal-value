# -*- coding: utf-8 -*-
"""
Spyder Danial Mohseni Taheri

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import math
import GammaProcess


class generate_Sample_Avr:
    def __init__(self):
        self.config = self.read_parameters_form_json()
        self.KX = self.config['KX']
        self.muY = self.config['muY']
        self.lambdaX = self.config['lambdaX']
        self.sigmaX = self.config["sigmaX"]
        self.sigmaY = self.config["sigmaY"]
        self.rhoXY = self.config["rhoXY"]
        self.dt = self.config['dt']
        self.beta = self.config['beta']
        self.alpha = self.config['alpha']
        self.InflowMeans = self.config["inflow_log_mean"]
        self.InflowSd = self.config["inflow_log_sd"]
        self.KInflow = self.config["KInflow"]
        self.sigmaInflow = self.config["sigmaInflow"]
        self.parameters = {'shape_param': self.config["shape_param"], 'rate_param':\
                  self.config["rate_param"]}
        
        
    def read_parameters_form_json(self):
        with open('Parameters.json') as f:
            config = json.load(f)
        return config
        
    def generate_Sample(self, I, K, initial_condition):
        T = self.config['nPeriods'] * self.config['dt']
        self.Inflow0 = initial_condition['Inflow0']
        self.X0 = initial_condition['X0']
        self.Y0 = initial_condition['Y0']
        
        def generate_loginf_logprice(T):
            dwX = np.random.normal(0, 1, I)
            dwY = np.random.normal(0, 1, I)
            dwInflow = np.random.normal(0, np.sqrt((1-np.exp(-2 * self.KInflow * self.dt))/\
                                             (2*self.KInflow)), I)
            Sigma = [(1-math.exp(-2*self.KX * self.dt))/(2*self.KX)*\
                     self.sigmaX**2, self.rhoXY*self.sigmaX*\
                     self.sigmaY*(1-math.exp(-self.KX*self.dt))/\
                     self.KX, self.rhoXY*self.sigmaX*\
                     self.sigmaY *(1-math.exp(-self.KX * self.dt))/\
                     self.KX, self.sigmaY**2*self.dt]

            Sigma = np.asarray(Sigma).reshape(2,2)
            S_chol = np.transpose(np.linalg.cholesky(Sigma))
            x = np.ones(I)* self.X0
            y = np.ones(I)* self.Y0
            Inflow = np.ones(I)* self.Inflow0
            for i in range(1, I):
                x[i] = x[i-1]* math.exp(-self.KX * self.dt)-\
                self.lambdaX*(1 - math.exp(-self.KX*self.dt))+\
                S_chol[0,0]*dwX[i-1]
                                        
                y[i] = y[i-1] + self.muY*self.dt + S_chol[1,0]*dwX[i-1]+\
                S_chol[1,1]*dwY[i-1]
                Inflow[i] = Inflow[i-1] * math.exp(-self.KInflow * self.dt)+\
                self.sigmaInflow * dwInflow[i-1]
            return x, y, Inflow

        #Store inflow (I) and price (P) on original scale
        w = np.zeros((I, K, 6))
        start_time_position = {"startTime": initial_condition['Start_time'], "startPosition": initial_condition['Gamma0']}
        Gamma = GammaProcess.Gamma_process(self.parameters, start_time_position)
        for k in range(K):
            x, y, Inflow = generate_loginf_logprice(T)
                                              
            for i in range(I):
                w[i, k, 0] = np.exp(self.alpha * np.cos(2*np.pi * i * self.dt)+\
                                    self.beta * np.sin(2* np.pi * i * self.dt))
                w[i, k, 1] = np.exp(x[i])
                w[i, k, 2] = np.exp(y[i])
                w[i, k, 3] = np.exp(self.config["inflow_log_mean"][i%52]) 
                w[i, k, 4] = np.exp(Inflow[i]* self.config["inflow_log_sd"][i%52])
                
            w[:, k, 5] = np.array(Gamma._generate_sample_path(range(I)))
                      
        return w
    
    def generate_Average(self, I, K, initial_condition):
        E_w = np.zeros((I, K, 6))
        time_space_constraints= {"startTime": initial_condition['Start_time'], "startPosition": initial_condition['Gamma0']}
        Gamma = GammaProcess.Gamma_process(self.parameters, time_space_constraints)
        Y0 = initial_condition['Y0'] 
        X0 = initial_condition['X0']
        Inflow0 = initial_condition['Inflow0']
        T_0 = initial_condition['Start_time']
        for i in range(I):
            price_longtermMean = Y0 + self.muY * i* self.dt
            price_shorttermMean = math.exp(-self.KX * i* self.dt)* X0 - \
                                        ((1-math.exp(-self.KX * i* self.dt))* self.lambdaX/self.KX)
            price_variance = (1-math.exp(-2 * self.KX * i* self.dt))*self.sigmaX**2/(2*self.KX) +\
            self.sigmaY**2 * i * self.dt  + 2* (1-math.exp(-(self.KX)* i * self.dt)) *\
            (self.sigmaX * self.sigmaY * self.rhoXY/(self.KX))
                                     
            price_constant = self.alpha * np.cos(2*np.pi * ((i+T_0) * self.dt))+\
            self.beta * np.sin(2* np.pi*((i+T_0) * self.dt))   
            
            Inflow_constant = self.InflowMeans[(i+T_0)%52]
            Inflow_Mean =  self.InflowSd[(i+T_0)%52] * math.exp(-self.KInflow * i * self.dt)*\
            Inflow0
            Inflow_Var = self.InflowSd[(i+T_0)%52]**2 * (1-math.exp(-2*self.KInflow*i* self.dt))*\
            self.sigmaInflow**2/(2*self.KInflow)
            if i == 0:
                print("Inside scenario, X_0", price_shorttermMean)
                print("Inside scenario, Y_0", price_longtermMean)
            for k in range(K):
                E_w[i, k, 0] = np.exp(price_constant + 0.5 * price_variance)
                E_w[i, k, 1] = np.exp(price_shorttermMean)
                E_w[i, k, 2] = np.exp(price_longtermMean)
                E_w[i, k, 3] = np.exp(Inflow_constant +  (1/2) * Inflow_Var)
                E_w[i, k, 4] = np.exp(Inflow_Mean)
                E_w[i, k, 5] = Gamma._get_mean_at(i)

        return E_w

def read_parameters_form_json():
    with open('Parameters.json') as f:
        config = json.load(f)
    return config

if __name__ == '__main__':
    config = read_parameters_form_json()
    #Inflow plot
    initial_condition = {'X0': config['X0'], 'Y0': config['Y0'],\
                    'Inflow0': config['Inflow0'], 'Start_time': 0, 'Gamma0': config['plant_cond0']}
    Sample = generate_Sample_Avr()
    K = config["n_Samples"]
    I = config["nPeriods"]
    w = Sample.generate_Sample(initial_condition)
    E_w = Sample.generate_Average(I, K, initial_condition)
    
    plt.figure()
    plt.plot(range(config['nPeriods']), np.multiply(E_w[:, :, 3], E_w[:, :, 4]) )
    #plt.plot(range(n), meanI.transpose())
    plt.xlabel("Time(week)")
    plt.ylabel("det. rate")
    plt.show()




    #Price plot
    plt.figure()
    plt.plot(range(config['nPeriods']), np.mean(np.multiply(w[:, :, 3], w[:, :, 4]), axis = 1), label = "emprical mean")
    plt.plot(range(config['nPeriods']), np.multiply(E_w[:, 1, 3], E_w[:, 1, 4]), label = "closed-form")
    plt.xlabel("Time(week)")
    plt.ylabel("det. rate")
    plt.legend()
    plt.show()
    #Price plot
    plt.figure()
    plt.plot(range(config['nPeriods']), np.multiply(w[:, :, 3], w[:, :, 4]))

    plt.xlabel("Time(week)")
    plt.ylabel("det. rate")
    plt.show()