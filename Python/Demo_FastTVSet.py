# -*- coding: utf-8 -*-
"""
@author: Mike

Test Fast-TV Set
"""

import torch 
#import time
from FastTVSet import *

g = torch.randint(low=0, high=255, size=(1,1,50, 50, 50)).float().cuda()
c = torch.randint(low=0, high=255, size=(1,1,50, 50, 50)).float()
Ld = 1
Lw = 1
Lh = 1
L = 1
NitTV = 10 #200
NitD = 3 #50

g = torch.randint(low=0, high=255, size=(1,1,50)).float().cuda()
g2 = FastTV_1D_GPU(g,L,NitTV)
print('Check: 1D-GPU')    

g = torch.randint(low=0, high=255, size=(1,1,50, 50)).float().cuda()
g2 = FastTV_2D_GPU(g,Lh,Lw,NitTV,NitD)
print('Check: 2D-GPU') 
    
g = torch.randint(low=0, high=255, size=(1,1,50, 50, 50)).float().cuda()
g2 = FastTV_3D_GPU(g,Ld,Lh,Lw,NitTV,NitD)
print('Check: 3D-GPU')    
    
g2 = FastTV_Directional3D_GPU(g,1,L,NitTV,NitD)
print('Check: 1 3D-GPU') 
g2 = FastTV_Directional3D_GPU(g,2,L,NitTV,NitD)
print('Check: 2 3D-GPU') 
g2 = FastTV_Directional3D_GPU(g,3,L,NitTV,NitD)
print('Check: 3 3D-GPU')    


# CPU set
c = torch.randint(low=0, high=255, size=(1,1,50)).float()     
c2 = FastTV_1D_CPU(c,L,NitTV)
print('Check: 1D-CPU')       

c = torch.randint(low=0, high=255, size=(1,1,50, 50)).float()
c2 = FastTV_2D_CPU(c,Lh,Lw,NitTV,NitD)
print('Check: 2D-CPU')      

c = torch.randint(low=0, high=255, size=(1,1,50, 50, 50)).float()
c2 = FastTV_3D_CPU(c,Ld,Lh,Lw,NitTV,NitD)
print('Check: 3D-CPU')     

c2 = FastTV_Directional3D_CPU(c,1,L,NitTV,NitD)
print('Check: 1 3D-CPU')     
c2 = FastTV_Directional3D_CPU(c,2,L,NitTV,NitD)
print('Check: 2 3D-CPU')   
c2 = FastTV_Directional3D_CPU(c,3,L,NitTV,NitD)
print('Check: 3 3D-CPU')   


