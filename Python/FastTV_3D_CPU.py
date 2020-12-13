# -*- coding: utf-8 -*-
"""
Fast Total Variation Denoising using Iterative Clipping Algorithm 
nested within Parallel Dykstra-like Proximal Algorithm.

Minimizing the cost function: F(X)= 1/2||Y-X||_2^2 + Ld||DdX||_1 + Lh||DhX||_1 + Lw||DwX||_1 

INPUT
  y1 - noisy signal 
  Ld - regularization parameter for the first dimension 
  Lh - regularization parameter for the second dimension 
  Lw - regularization parameter for the third dimension 
  NitTV - number of iterations to run the Iterative Clipping Algorithm
  NitD - number of iterations to run the Parallel Dykstra-like Proximal Algorithm

OUTPUT
  Xdn - denoised signal

Reference
'Fast Speckle Noise Reduction For OCT  Imaging', Michael Shamouilian, NYU Dissertation, 2021.


"""


import torch
#import numpy as np

def FastTV_3D_CPU(y1,Ld,Lh,Lw,NitTV,NitD):
    
    [n,c,d,h,w] = y1.shape
    
    y2 = y1
    y3 = y1
    
    # Create filters oriented in each of the 3 directions
    # Filters created directly on the GPU
    h1a = torch.zeros((1,1,1,2,1)).float()
    h1a[0,0,0,0,0] = 1
    h1a[0,0,0,1,0] = -1
    h2a = torch.zeros((1,1,1,3,1)).float()
    h2a[0,0,0,0,0] = .25
    h2a[0,0,0,1,0] = .5
    h2a[0,0,0,2,0] = .25
    
    h1b = torch.zeros((1,1,2,1,1)).float()
    h1b[0,0,0,0,0] = 1
    h1b[0,0,1,0,0] = -1
    h2b = torch.zeros((1,1,3,1,1)).float()
    h2b[0,0,0,0,0] = .25
    h2b[0,0,1,0,0] = .5
    h2b[0,0,2,0,0] = .25
    
    h1c = torch.zeros((1,1,1,1,2)).float()
    h1c[0,0,0,0,0] = 1
    h1c[0,0,0,0,1] = -1
    h2c = torch.zeros((1,1,1,1,3)).float()
    h2c[0,0,0,0,0] = .25
    h2c[0,0,0,0,1] = .5
    h2c[0,0,0,0,2] = .25
    
    # Outer loop - Parallel Dykstra-like Proximal Algorithm
    for i in range(NitD):
        
        # Iterative Clipping Algorithm in the h-direction
        z = torch.zeros((n,c,d,h-1,w)).float()
        bias = torch.nn.functional.conv3d(y1,h1a)/4
        for i in range(NitTV):
            z = torch.clamp((torch.nn.functional.conv3d(z,h2a,padding=(0,1,0))+bias), min=(-1*Lh), max=Lh)
        p1 = y1-torch.nn.functional.conv3d(z,-1*h1a,padding=(0,1,0))
        #torch.cuda.empty_cache()
        
        # Iterative Clipping Algorithm in the d-direction
        z = torch.zeros((n,c,d-1,h,w)).float()
        bias = torch.nn.functional.conv3d(y2,h1b)/4
        for i in range(NitTV):
            z = torch.clamp((torch.nn.functional.conv3d(z,h2b,padding=(1,0,0))+bias), min=(-1*Ld), max=Ld)
        p2 = y2-torch.nn.functional.conv3d(z,-1*h1b,padding=(1,0,0))
        #torch.cuda.empty_cache()
        
        # Iterative Clipping Algorithm in the w-direction
        z = torch.zeros((n,c,d,h,w-1)).float()
        bias = torch.nn.functional.conv3d(y3,h1c)/4
        for i in range(NitTV):
            z = torch.clamp((torch.nn.functional.conv3d(z,h2c,padding=(0,0,1))+bias), min=(-1*Lw), max=Lw)
        p3 = y3-torch.nn.functional.conv3d(z,-1*h1c,padding=(0,0,1))
        #torch.cuda.empty_cache()
        
        # Each direction weighted equally in the Dykstra Algorithm
        Xdn = (p1+p2+p3)/3
        
        # Update output of Dykstra Algorithm
        y1 = Xdn+y1-p1
        y2 = Xdn+y2-p2
        y3 = Xdn+y3-p3
        
    return Xdn