# FastTVD
Fast Total Variation Denoising and Speckle Denoising Utilizing GPUs and Multi-Core CPUs

Michael Shamouilian, NYU, 2020

Fast Total Variation Denoising using Iterative Clipping Algorithm nested within Parallel Dykstra-like Proximal Algorithm for multi-dim data.

Optimized for use with GPUs and multicore CPUs

Solving the problem of minimizing the 3D TVD cost function: F(X)= 1/2||Y-X||_2^2 + Li||DiX||_1 + Lj||DjX||_1 + Lk||DkX||_1 

Solving the problem of minimizing the 2D TVD cost function: F(X)= 1/2||Y-X||_2^2 + Li||DiX||_1 + Lj||DjX||_1 

Solving the problem of minimizing the 1D TVD cost function: F(X)= 1/2||Y-X||_2^2 + Li||DiX||_1 

Note: That Reduced Directional 3D TVD is used to perform TVD on 3D in a single direction, or to perform 1D TVD on multiple 1D signals clustered in 3D. 

If you use our code for your project, please cite: 'Fast Speckle Noise Reduction For OCT  Imaging', Michael Shamouilian, NYU Dissertation, 2021.


4 separate functions are provided for 4 sets of systems. There are python and matlab versions, with and without GPU acceleration. 
We provide an all inclusive code that allows for the use of 1D, 2D, 3D and Reduced Directional 3D TVD. 
Note: for GPU use, a CUDA enabled GPU is necessary. 
Note: for python use, Pytorch is required. Install Pytorch from https://pytorch.org/

Note: FastTVSet.py contains all of the python functions to allow for easy import and use. 

Demo_FastTVSet.py shows an example of how to use all the python functions.

Demo_FastSet.m shows an example of how to use all the matlab functions.

