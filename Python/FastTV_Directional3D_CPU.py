# -*- coding: utf-8 -*-
"""
Fast Total Variation Denoising using Iterative Clipping Algorithm 
nested within Parallel Dykstra-like Proximal Algorithm.

Optimized for use with GPUs

Minimizing the cost function: F(X)= 1/2||Y-X||_2^2 + Ld||DdX||_1 

INPUT
  y1 - noisy signal
  ddim - dimension to perform TVD in the direction of (Scalar: 1,2,3)
  L - TV regularization parameter  
  NitTV - number of iterations to run the Iterative Clipping Algorithm
  NitD - number of iterations to run the Parallel Dykstra-like Proximal Algorithm

OUTPUT
  Xdn - denoised signal

Reference
'Fast Speckle Noise Reduction For OCT  Imaging', Michael Shamouilian, NYU Dissertation, 2021.

"""


import torch
#import numpy as np

def FastTV_Directional3D_CPU(y1,ddim,L,NitTV,NitD):
    
    [n,c,d,h,w] = y1.shape

    
    if ddim == 1:
        
        h1a = torch.zeros((1,1,1,2,1)).float()
        h1a[0,0,0,0,0] = 1
        h1a[0,0,0,1,0] = -1
        h2a = torch.zeros((1,1,1,3,1)).float()
        h2a[0,0,0,0,0] = .25
        h2a[0,0,0,1,0] = .5
        h2a[0,0,0,2,0] = .25
        # Iterative Clipping Algorithm in the h-direction
        z = torch.zeros((n,c,d,h-1,w)).float()
        bias = torch.nn.functional.conv3d(y1,h1a)/4
        for i in range(NitTV):
            z = torch.clamp((torch.nn.functional.conv3d(z,h2a,padding=(0,1,0))+bias), min=(-1*L), max=L)
        Xdn = y1-torch.nn.functional.conv3d(z,-1*h1a,padding=(0,1,0))
        #torch.cuda.empty_cache()
        return Xdn
    
    elif ddim == 2: 
        
        h1b = torch.zeros((1,1,2,1,1)).float()
        h1b[0,0,0,0,0] = 1
        h1b[0,0,1,0,0] = -1
        h2b = torch.zeros((1,1,3,1,1)).float()
        h2b[0,0,0,0,0] = .25
        h2b[0,0,1,0,0] = .5
        h2b[0,0,2,0,0] = .25
        # Iterative Clipping Algorithm in the d-direction
        z = torch.zeros((n,c,d-1,h,w)).float()
        bias = torch.nn.functional.conv3d(y1,h1b)/4
        for i in range(NitTV):
            z = torch.clamp((torch.nn.functional.conv3d(z,h2b,padding=(1,0,0))+bias), min=(-1*L), max=L)
        Xdn = y1-torch.nn.functional.conv3d(z,-1*h1b,padding=(1,0,0))
        #torch.cuda.empty_cache()
        return Xdn
    
    else:    
        
        h1c = torch.zeros((1,1,1,1,2)).float()
        h1c[0,0,0,0,0] = 1
        h1c[0,0,0,0,1] = -1
        h2c = torch.zeros((1,1,1,1,3)).float()
        h2c[0,0,0,0,0] = .25
        h2c[0,0,0,0,1] = .5
        h2c[0,0,0,0,2] = .25
        # Iterative Clipping Algorithm in the w-direction
        z = torch.zeros((n,c,d,h,w-1)).float()
        bias = torch.nn.functional.conv3d(y1,h1c)/4
        for i in range(NitTV):
            z = torch.clamp((torch.nn.functional.conv3d(z,h2c,padding=(0,0,1))+bias), min=(-1*L), max=L)
        Xdn = y1-torch.nn.functional.conv3d(z,-1*h1c,padding=(0,0,1))
        #torch.cuda.empty_cache()
        return Xdn