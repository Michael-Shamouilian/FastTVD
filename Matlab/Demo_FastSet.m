% Fast Total Variation Denoising using Iterative Clipping Algorithm 
% nested within Parallel Dykstra-like Proximal Algorithm.
% 
% Optimized for parallel processing
% 
% Minimizing the cost function: F(X)= 1/2||Y-X||_2^2 + TV(X) 
% 
% Demo Code
% 
% Reference
% 'Fast Speckle Noise Reduction For OCT  Imaging', Michael Shamouilian, NYU Dissertation, 2021.
% 


NitTV = 50;
NitD = 10;
L = 1;

% Test GPU Methods
Y = gpuArray(randi(255,[1,128]));
X = FastTV_1D_GPU(Y,L,NitTV);

Y = gpuArray(randi(255,[128,128]));
X = FastTV_2D_GPU(Y,L,L,NitTV,NitD);

Y = gpuArray(randi(255,[128,128,128]));
X = FastTV_3D_GPU(Y,L,L,L,NitTV,NitD);
X = FastTV_Directional3D_GPU(Y,1,L,NitTV);
X = FastTV_Directional3D_GPU(Y,2,L,NitTV);
X = FastTV_Directional3D_GPU(Y,3,L,NitTV);


% Test CPU Methods
Y = randi(255,[1,128]);
X = FastTV_1D_CPU(Y,L,NitTV);

Y = randi(255,[128,128]);
X = FastTV_2D_CPU(Y,L,L,NitTV,NitD);

Y = randi(255,[128,128,128]);
X = FastTV_3D_CPU(Y,L,L,L,NitTV,NitD);
X = FastTV_Directional3D_CPU(Y,1,L,NitTV);
X = FastTV_Directional3D_CPU(Y,2,L,NitTV);
X = FastTV_Directional3D_CPU(Y,3,L,NitTV);