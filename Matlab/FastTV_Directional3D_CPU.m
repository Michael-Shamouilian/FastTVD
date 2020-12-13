function [X] = FastTV_Directional3D_CPU(Y,ddim,Lam,NitTV)
% 
% [x, cost] = tvd_mm(y, lam, Nit)
% Fast Total Variation Denoising using Iterative Clipping Algorithm 
% nested within Parallel Dykstra-like Proximal Algorithm.
%
% Perform TVD in one direction on 3D data or Multiple 1D vectors in parallel 
% 
% Minimizing the cost function: F(X)= 1/2||Y-X||_2^2 + Li||DiX||_1 
%
% INPUT
%   Y - noisy signal
%   ddim - dimension to perform TVD in the direction of (Scalar: 1,2,3)
%   Lam - regularization parameter for the first dimension (may be scalar or size(Y))
%   NitTV - number of iterations to run the Iterative Clipping Algorithm
%
% OUTPUT
%   X - denoised signal
%
% Reference
% 'Fast Speckle Noise Reduction For OCT  Imaging', Michael Shamouilian, NYU Dissertation, 2021.
% 


[r,c,d] = size(Y);

switch ddim
    case 1
        
        h1b = ones(2,1,1); h1b(1) = 1; h1b(2)=-1;
        h2b = ones(3,1,1); h2b(1) = .25; h2b(2)=.5; h2b(3)=.25;
        
        % Iterative Clipping Algorithm in the i-direction
        z = zeros(r-1,c,d);
        bias = convn(Y,h1b,'valid')/4;
        for n=1:NitTV
            z = max(min(convn(z,h2b,'same')+bias,Lam),-Lam);
        end
        X = Y-convn(z,-h1b);
        
    case 2
        
        h1a = ones(1,2,1); h1a(1) = 1; h1a(2)=-1;
        h2a = ones(1,3,1); h2a(1) = .25; h2a(2)=.5; h2a(3)=.25; 
        
        % Iterative Clipping Algorithm in the j-direction
        z = zeros(r,c-1,d);
        bias = convn(Y,h1a,'valid')/4;
        for n=1:NitTV
            z = max(min(convn(z,h2a,'same')+bias,Lam),-Lam);
        end
        X = Y-convn(z,-h1a);
        
    case 3
        
        h1c = ones(1,1,2); h1c(1) = 1; h1c(2)=-1;
        h2c = ones(1,1,3); h2c(1) = .25; h2c(2)=.5; h2c(3)=.25;
        
        % Iterative Clipping Algorithm in the k-direction
        z = zeros(r,c,d-1);
        bias = convn(Y,h1c,'valid')/4;
        for n=1:NitTV
            z = max(min(convn(z,h2c,'same')+bias,Lam),-Lam);
        end
        X = Y-convn(z,-h1c);
        
    otherwise
        
        disp('Eneter valid direction (1,2,3)')

end    

