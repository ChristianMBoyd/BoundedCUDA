// CUDA file to organize GPU functions, kernel calls, etc.
#pragma once
// CUDA prereqs
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h> // << < >> > usages
// std->CUDA math libraries
#include <cuda/std/cmath>
#include <cuda/std/complex>
// cooperative group kernel methods
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

// simplify cooperative group calls
namespace cg = cooperative_groups;

// device function declarations

__device__ cuda::std::complex<double> posRoot(cuda::std::complex<double> arg); // test code
__device__ cuda::std::complex<double> Pi0Qn(double q, double w, double delta, double Qn, double Qnp);
__device__ cuda::std::complex<double> Pi0Qnp(double q, double w, double delta, double Qn, double Qnp);
__device__ void sumBlock(cuda::std::complex<double>* sData, const cg::thread_block& block);
__device__ void sumPi0Qn(const cg::thread_block& block, cuda::std::complex<double>* sData,
	double q, double w, double delta, double dQ, double Qn, int min, const int tot);
__device__ void sumPi0Qnp(const cg::thread_block& block, cuda::std::complex<double>* sData,
	double q, double w, double delta, double dQ, double Qn, double L);

// kernel declarations

__global__ void posRootKernel(const cuda::std::complex<double>* arg, cuda::std::complex<double>* root); // test of complex math
__global__ void mChi0DiagKernel(const double* QList, cuda::std::complex<double>* mChi0Diag, int size,
	double q, double w, double delta, double L, bool evenPar);
