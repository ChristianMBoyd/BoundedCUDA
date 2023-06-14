#pragma once
// CUDA prereqs
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// std math CPU libraries
#include <stdio.h>
#include <cmath>
#include <complex>
// std->CUDA math libraries
#include <cuda/std/cmath>
#include <cuda/std/complex>

class Loss
{
public:
	Loss();
	void initializeDevice();
	cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);
	__global__ void addKernel(int* c, const int* a, const int* b);

	// purely test functions:
	void Hello();
}