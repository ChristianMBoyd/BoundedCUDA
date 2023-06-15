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

	// purely test functions
	cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size); // contains a lot of CUDA example usage
	cudaError_t sqrtWithCuda(const double* arg, double* root, unsigned int size); // sqrt test
	// posRoot test
	cudaError_t posRootWithCuda(const cuda::std::complex<double>* arg, cuda::std::complex<double>* root, unsigned int size);
};