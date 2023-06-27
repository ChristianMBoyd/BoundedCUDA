#pragma once
//CPU libraries
#include <stdio.h> // CUDA-provided i/o
#include <iostream> // preferred i/o
#include <cmath> // standard CPU math
// CUDA libraries
#include <cuda_runtime.h>
#include <cuda_runtime_api.h> // generic kernel calls 
#include <device_launch_parameters.h> // TBD if needed
// std->CUDA math libraries
#include <cuda/std/cmath>
#include <cuda/std/complex>

class Loss
{
public:
	Loss();

	// GPU management
	cudaDeviceProp deviceProp;
	int MAX_BLOCKS_PER_MP;
	int MAX_THREADS_PER_MP;
	int MAX_THREADS_PER_BLOCK;
	int NUM_MP;
	cudaError_t initializeDevice();
	void deviceQuery();
	bool check(cudaError_t status, const char* errorReport);
	bool check(cudaError_t status, const char* errorReport, const char* cudaErrorString);
	bool check(cudaError_t status, const char* errorReport, cudaError_t errorCode);

	// calculations and kernel calls
	cudaError_t posRootWithCuda(const void* posRootKernel, const cuda::std::complex<double>* arg,
		cuda::std::complex<double>* root, unsigned int size);
};