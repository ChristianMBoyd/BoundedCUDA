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
__device__ cuda::std::complex<double> posRoot(cuda::std::complex<double> arg);
__device__ cuda::std::complex<double> Pi0Qn(double q, double w, double delta, double Qn, double Qnp);
__device__ cuda::std::complex<double> Pi0Qnp(double q, double w, double delta, double Qn, double Qnp);

// kernel function and call declarations -- currently testing if calls can be removed
//void addKernel_call(int* c, const int* a, const int* b, unsigned int size); // external call to initiate kernel with size threads
//__global__ void addKernel(int* c, const int* a, const int* b);
//void sqrtKernel_call(const double* arg, double* root, unsigned int size);
//__global__ void sqrtKernel(const double* arg, double* root);
void posRootKernel_call(const cuda::std::complex<double>* arg, cuda::std::complex<double>* root, unsigned int size);
__global__ void posRootKernel(const cuda::std::complex<double>* arg, cuda::std::complex<double>* root);