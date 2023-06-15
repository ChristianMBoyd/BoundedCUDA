#include "Loss.h"

Loss::Loss()
{
    initializeDevice();
}

void Loss::initializeDevice()
{
    // set device for CUDA use
    cudaError_t status = cudaSetDevice(0);
    if (status != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Check GPU configuration.");
    }
}


// TEST CODE BELOW 

// basic vector addition
cudaError_t Loss::addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel_call(dev_c, dev_a, dev_b, size);

    // keeping error-checking outside of gpu.cu
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error: // failsafe: on error -> cleanup
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

// testing sqrt() functionality
cudaError_t Loss::sqrtWithCuda(const double* arg, double* root, unsigned int size)
{
    // define CUDA pointers and error-checking
    double* dev_arg = 0;
    double* dev_root = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for two vector (one input, one output)
    cudaStatus = cudaMalloc((void**)&dev_arg, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error; // trigger cleanup on failure
    }

    cudaStatus = cudaMalloc((void**)&dev_root, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    // Copy input vector from host to device memory: cudaMemcpyHostToDevice flag
    cudaStatus = cudaMemcpy(dev_arg, arg, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    sqrtKernel_call(dev_arg, dev_root, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rootKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory - cudaMemcpyDeviceToHost flag
    cudaStatus = cudaMemcpy(root, dev_root, size * sizeof(double), cudaMemcpyDeviceToHost); 
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error: // failsafe: on error -> cleanup
    cudaFree(dev_arg);
    cudaFree(dev_root);

    return cudaStatus;
}

// testing posRoot functionality
cudaError_t Loss::posRootWithCuda(const cuda::std::complex<double>* arg, cuda::std::complex<double>* root, unsigned int size)
{
    // define CUDA pointers and error-checking
    cuda::std::complex<double>* dev_arg = 0; // is it always efficient to zero out values first?
    cuda::std::complex<double>* dev_root = 0;
    cudaError_t cudaStatus;

    // Allocate GPU buffers for two vector (one input, one output)
    cudaStatus = cudaMalloc((void**)&dev_arg, size * sizeof(cuda::std::complex<double>));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error; // trigger cleanup on failure
    }

    cudaStatus = cudaMalloc((void**)&dev_root, size * sizeof(cuda::std::complex<double>));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    // Copy input vector from host to device memory: cudaMemcpyHostToDevice flag
    cudaStatus = cudaMemcpy(dev_arg, arg, size * sizeof(cuda::std::complex<double>), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    posRootKernel_call(dev_arg, dev_root, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rootKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory - cudaMemcpyDeviceToHost flag
    cudaStatus = cudaMemcpy(root, dev_root, size * sizeof(cuda::std::complex<double>), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error: // failsafe: on error -> cleanup
    cudaFree(dev_arg);
    cudaFree(dev_root);

    return cudaStatus;
}