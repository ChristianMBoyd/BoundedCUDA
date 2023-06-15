// CUDA file to organize GPU functions, kernel calls, etc.
#pragma once
// CUDA prereqs
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// std->CUDA math libraries
#include <cuda/std/cmath>
#include <cuda/std/complex>

// function declarations

// test functions
void addKernel_call(int* c, const int* a, const int* b, unsigned int size); // external call to initiate kernel with size threads
__global__ void addKernel(int* c, const int* a, const int* b);
void sqrtKernel_call(const double* arg, double* root, unsigned int size);
__global__ void sqrtKernel(const double* arg, double* root);
void posRootKernel_call(const cuda::std::complex<double>* arg, cuda::std::complex<double>* root, unsigned int size);
__global__ void posRootKernel(const cuda::std::complex<double>* arg, cuda::std::complex<double>* root);