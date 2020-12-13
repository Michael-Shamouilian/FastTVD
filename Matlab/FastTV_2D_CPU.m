function [X] = FastTV_2D_CPU(Y,Li,Lj,NitTV,NitD)
% 
% [x, cost] = tvd_mm(y, lam, Nit)
% Fast Total Variation Denoising using Iterative Clipping Algorithm 
% nested within Parallel Dykstra-like Proximal Algorithm.
% 
% Minimizing the cost function: F(X)= 1/2||Y-X||_2^2 + Li||DiX||_1 + Lj||DjX||_1
%
% INPUT
%   Y - noisy signal 
%   Li - regularization parameter for the first dimension (may be scalar or size(Y))
%   Lj - regularization parameter for the second dimension (may be scalar or size(Y))
%   NitTV - number of iterations to run the Iterative Clipping Algorithm
%   NitD - number of iterations to run the Parallel Dykstra-like Proximal Algorithm
%
% OUTPUT
%   X - denoised signal
%
% Reference
% 'Fast Speckle Noise Reduction For OCT  Imaging', Michael Shamouilian, NYU Dissertation, 2021.
% 

[r,c,d] = size(Y);
y2 = Y;

% Create filters oriented in each of the 3 directions
% Filters created directly on the GPU
h1a = ones(1,2,1); h1a(1) = 1; h1a(2)=-1;
h2a = ones(1,3,1); h2a(1) = .25; h2a(2)=.5; h2a(3)=.25; 

h1b = ones(2,1,1); h1b(1) = 1; h1b(2)=-1;
h2b = ones(3,1,1); h2b(1) = .25; h2b(2)=.5; h2b(3)=.25; 


% Outer loop - Parallel Dykstra-like Proximal Algorithm
for i = 1:NitD
    
    % Iterative Clipping Algorithm in the j-direction
    z = zeros(r,c-1,d);
    bias = convn(Y,h1a,'valid')/4;
    for n=1:NitTV
        z = max(min(convn(z,h2a,'same')+bias,Lj),-Lj);
    end
    p1 = Y-convn(z,-h1a);
    
    % Iterative Clipping Algorithm in the i-direction
    z = zeros(r-1,c,d);
    bias = convn(y2,h1b,'valid')/4;
    for n=1:NitTV
        z = max(min(convn(z,h2b,'same')+bias,Li),-Li);
    end
    p2 = y2-convn(z,-h1b);
    
    
    % Each direction weighted equally in the Dykstra Algorithm 
    X = (p1+p2)/2;
    
    % Update output of Dykstra Algorithm
    Y = X+Y-p1;
    y2 = X+y2-p2;
    
end
end

