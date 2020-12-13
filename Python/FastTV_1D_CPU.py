# -*- coding: utf-8 -*-
"""
Fast Total Variation Denoising using Iterative Clipping Algorithm 
nested within Parallel Dykstra-like Proximal Algorithm.

Optimized for use with GPUs

Minimizing the cost function: F(X)= 1/2||Y-X||_2^2 + L||DX||_1 

INPUT
  y1 - noisy signal 
  L - TV regularization parameter  
  NitTV - number of iterations to run the Iterative Clipping Algorithm

OUTPUT
  Xdn - denoised signal

Reference
'Fast Speckle Noise Reduction For OCT  Imaging', Michael Shamouilian, NYU Dissertation, 2021.


"""


import torch
#import numpy as np

def FastTV_1D_CPU(y1,L,NitTV):
    
    [n,c,l] = y1.shape
    
    
    # Create filters oriented in each of the 3 directions
    # Filters created directly on the GPU
    h1a = torch.zeros((1,1,2)).float()
    h1a[0,0,0] = 1
    h1a[0,0,1] = -1
    h2a = torch.zeros((1,1,3)).float()
    h2a[0,0,0] = .25
    h2a[0,0,1] = .5
    h2a[0,0,2] = .25
    
        
    # Iterative Clipping Algorithm in the h-direction
    z = torch.zeros((n,c,l-1)).float()
    bias = torch.nn.functional.conv1d(y1,h1a)/4
    for i in range(NitTV):
        z = torch.clamp((torch.nn.functional.conv1d(z,h2a,padding=(1))+bias), min=(-1*L), max=L)
    Xdn = y1-torch.nn.functional.conv1d(z,-1*h1a,padding=(1))
    #torch.cuda.empty_cache()
        
        
    return Xdn