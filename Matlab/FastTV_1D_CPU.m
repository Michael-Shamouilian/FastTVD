function [X] = FastTV_1D_CPU(Y,Lam,Nit)
% 
% [x, cost] = tvd_mm(y, lam, Nit)
% Fast Total Variation Denoising using Iterative Clipping Algorithm 
% nested within Parallel Dykstra-like Proximal Algorithm.
% 
% Minimizing the cost function: F(X)= 1/2||Y-X||_2^2 + Lam||DiX||_1  
%
% INPUT
%   Y - noisy signal 
%   Lam - TV regularization parameter (may be scalar or size(Y))
%   Nit - number of iterations 
%
% OUTPUT
%   X - denoised signal
%
% Reference
% 'Fast Speckle Noise Reduction For OCT  Imaging', Michael Shamouilian, NYU Dissertation, 2021.
% 


N = length(Y);
       
T = Lam/2;

% Create Filters
h1a = [1,-1];
h2a = [.25,.5,.25];

% Iterative Clipping Algorithm
z = zeros(1,N-1);
bias = convn(Y,h1a,'valid')/4; 
for k=1:Nit
    z = max(min(convn(z,h2a,'same')+bias,T),-T);       
end
X = Y-convn(z,-h1a);

end
