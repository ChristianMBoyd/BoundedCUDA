#include "Loss.cuh"

Loss::Loss()
{
	initializeDevice();
}

__global__ void Loss::initializeDevice()
{
	// set device for CUDA use
	cudaError_t status = cudaSetDevice(0);
	if (status != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Check GPU configuration.");
	}
}

void Loss::Hello()
{
	printf("Hello");
}

__global__ void Loss::addKernel(int* c, const int* a, const int* b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}