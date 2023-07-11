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

	//GPU management

	cudaDeviceProp deviceProp;
	int MAX_BLOCKS_PER_MP, MAX_THREADS_PER_MP, MAX_THREADS_PER_BLOCK, NUM_MP,
		MAX_SMEM_PER_BLOCK, MAX_SMEM_PER_MP;
	cudaError_t initializeDevice();
	void deviceQuery();
	void check(bool& working, cudaError_t status, const char* errorReport);
	void check(bool& working, cudaError_t status, const char* errorReport, const char* cudaErrorString);
	void check(bool& working, cudaError_t status, const char* errorReport, cudaError_t errorCode);
	cudaError_t optimizeLaunchParameters(int& threadsPerBlock, int& blocksPerGrid, int& smemSize,
		const int threadSmem, const void* kernelFunc);
	// calculations

	bool evenQ(const int val);
	int nMax(const double cutoff, const double L);
	int totNum(const int nMax, const bool evenPar);
	void initializeQList(double Qlist[], const double L, const int tot, const bool evenPar);

	// kernel calls

	cudaError_t mChi0DiagLaunch(const void* mChi0DiagKernel, const double QList[], cuda::std::complex<double> mChi0Diag[],
		int size, double q, double w, double delta, double L, bool evenPar);
	cudaError_t posRootWithCuda(const void* posRootKernel, const cuda::std::complex<double>* arg,
		cuda::std::complex<double>* root, unsigned int size);

	// constants

	const double pi = 3.141592653589793238463;
};