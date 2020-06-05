import torch
import random
import math
import numpy as np


def InitializeParameters():

    ## Initializing all the parameters
    #f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11 = [1000, 200], [1024, 200], [1024, 256], [1024], [1024], [1024, 256], [1024, 256], [1024], [1024], [1000, 256], [1000]
    #f1, f2, f3, f4, f5, f6, f7 = [1000, 200], [1024, 200], [1024, 256], [1024], [1024], [1000, 256], [1000]

    f1, f2, f3, f4, f5, f6, f7 = [1000, 200], [1024, 200], [1024, 256], [1024], [1024], [1000, 256], [1000]
    #init embedding
    f1 = initializeEmbedding(f1)

    #init lstm
    f2 = initializeLSTM(f2)
    f3 = initializeLSTM(f3)
    f4 = initializeLSTM(f4)
    f5 = initializeLSTM(f5)
    #f6 = initializeLSTM(f6)
    #f7 = initializeLSTM(f7)
    #f8 = initializeLSTM(f8)
    #f9 = initializeLSTM(f9)

    #init linear
    f6 = initializeLinear(f6)
    f7 = initializeLinear(f7)

    #params = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]
    params = [f1, f2, f3, f4, f5, f6, f7]
    return params

def initializeEmbedding(size):
    return torch.randn(size)

def initializeLSTM(size):
    s = math.sqrt(1.0 / 256)  #hidden_size=256
    return torch.from_numpy(np.random.uniform(-s, s, size))

def initializeLinear(size):
    s = math.sqrt(1.0 / 256)  #in_feature=256
    return torch.from_numpy(np.random.uniform(-s, s, size))
