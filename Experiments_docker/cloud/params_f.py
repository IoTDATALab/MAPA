import numpy as np
import gzip
import torch
f = 5
img_depth = 1
num_filt1 = 10
num_filt2 = 10

def InitializeParameters():

    ## Initializing all the parameters
    f1, f2, w3 = (num_filt1 ,img_depth,f,f), (num_filt2 ,num_filt1,f,f), [62, 160]
    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    w3 = initializeWeight(w3)
   

    b1 = torch.from_numpy(np.zeros(f1.shape[0]))
    b2 = torch.from_numpy(np.zeros(f2.shape[0]))
    b3 = torch.from_numpy(np.zeros(w3.shape[0]))
   

    params = [f1, b1,f2, b2, w3, b3]
    return params

def initializeFilter(size, scale = 1.0):
    
    stddev = scale/np.sqrt(np.prod(size))
    return torch.from_numpy(np.random.normal(loc = 0, scale = stddev, size = size))

def initializeWeight(size):
    return torch.from_numpy((np.random.standard_normal(size=size) * 0.01))