// CUDA file to organize GPU functions, kernel calls, etc.
#pragma once
// CUDA prereqs
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// std->CUDA math libraries
#include <cuda/std/cmath>
#include <cuda/std/complex>

// function declarations
void addKernel_call(unsigned int size, int* c, const int* a, const int* b); // external call to initiate kernel with size threads
__global__ void addKernel(int* c, const int* a, const int* b);