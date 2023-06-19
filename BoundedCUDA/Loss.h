#pragma once
//CPU libraries
#include <stdio.h> // CUDA-provided i/o
#include <iostream> // preferred i/o
#include <cmath> // standard CPU math
#include <vector> // current buffer class
// device code, kernel calls, and CUDA libraries
#include "gpu.cuh"

class Loss
{
public:
	Loss();

	// GPU management
	cudaDeviceProp deviceProp;
	cudaError_t initializeDevice();
	void deviceQuery();
	bool check(cudaError_t status, const char* errorReport);
	bool check(cudaError_t status, const char* errorReport, const char* cudaErrorString);
	bool check(cudaError_t status, const char* errorReport, cudaError_t errorCode);

	// purely test functions
	cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size); // contains a lot of CUDA example usage
	cudaError_t sqrtWithCuda(const double* arg, double* root, unsigned int size); // sqrt test

	// posRoot test
	cudaError_t posRootWithCuda(const cuda::std::complex<double>* arg, cuda::std::complex<double>* root, unsigned int size);
};