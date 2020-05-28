import numpy as np
import scipy as sp
import scipy.stats as stats
from scipy.special import gammainc
import warnings
import matplotlib.pyplot as plt

class Jump_Diffusion_process(object):
    def __init__(self, dict):    
        print("dict", dict)
        self.startTime = dict["startTime"]
        self.startPosition = dict["startPosition"]
        self.conditional=False

class Gamma_process(Jump_Diffusion_process):
    def __init__(self, parameters, time_space_constraints):
        super(Gamma_process,self).__init__(time_space_constraints)

        self.shape_param = parameters["shape_param"]
        self.rate_param = parameters["rate_param"]
        self.gamma = stats.gamma
    # gamma distribution scale = 1/rate
    def _generate_position_at(self,t):
        if not self.conditional:
            return self.gamma.rvs(a = self.shape_param /52. ,scale =\
                                                       1./self.rate_param)
        else:
            raise ValueError('condition should not exist.')
            
    def _get_mean_at(self,t):
        "notice the conditional is independent of self.mean."
        if self.conditional:
            raise ValueError('condition should not exist.')
        else:
            return self.shape_param /(self.rate_param*52.) 
    
    def _generate_sample_path(self,times):
        if not self.conditional:
            t = self.startTime
            x = self.startPosition
            g = 0
            path=[]
            for time in times:
                delta = (time - t)/52.
                try:
                    g = self.gamma.rvs(a = self.shape_param * delta, scale = 1./self.rate_param)
                except ValueError:
                    pass
                t = time
                path.append(g)
            return path
        else:
            raise ValueError('condition should not exist.')
