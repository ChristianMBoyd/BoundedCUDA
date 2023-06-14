#pragma once
// std math CPU libraries
#include <stdio.h>
#include <cmath>
#include <complex>
// link to CUDA interactions
#include "gpu.cuh"

class Loss
{
public:
	Loss();
	void initializeDevice();
	cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

	// purely test functions:
	void Hello();
};