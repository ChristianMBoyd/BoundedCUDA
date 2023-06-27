#include "Loss.h"

Loss::Loss()
{
    cudaError_t cudaStatus = initializeDevice();
    check(cudaStatus, "initializeDevice failed!");
}

// assumes single GPU at device "0," extracts useful info into deviceProp and sets device for subsequent use
cudaError_t Loss::initializeDevice()
{
    // error checking
    cudaError_t cudaStatus;
    bool working;

    // extract basic information from device
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    MAX_BLOCKS_PER_MP = deviceProp.maxBlocksPerMultiProcessor;
    MAX_THREADS_PER_BLOCK = deviceProp.maxThreadsPerBlock;
    NUM_MP = deviceProp.multiProcessorCount;
    MAX_THREADS_PER_MP = deviceProp.maxThreadsPerMultiProcessor;

    working = check(cudaStatus,"cudaGetDeviceProperties failed! Check GPU configuration.");
    
    if (working)
    {
        // set device for CUDA use
        cudaStatus = cudaSetDevice(0);
    }

    return cudaStatus;
}

// print basic GPU information
void Loss::deviceQuery()
{
    // assuming a single CUDA-enabled device and deviceProp initialized in initializeDevice()
    std::cout << "Using an " << deviceProp.name
        << " with compute capability " << deviceProp.major
        << "." << deviceProp.minor << "." << std::endl;
    std::cout << "Multiprocessor count: " << NUM_MP << "." << std::endl;
    std::cout << "Max threads per multiprocessor: " << MAX_THREADS_PER_MP << "." << std::endl;
    std::cout << "Max blocks per multiprocessor: " << MAX_BLOCKS_PER_MP << "." << std::endl;
    std::cout << "Max threads per block: " << MAX_THREADS_PER_BLOCK << "." << std::endl;
    std::cout << "'1' if concurrent kernels: " << deviceProp.concurrentKernels
        << "." << std::endl;
    std::cout << ">0 if async host-device memory transfer + kernel execution: "
        << deviceProp.asyncEngineCount << "." << std::endl;
    std::cout << "'1' if cooperative launch support: " << deviceProp.cooperativeLaunch
        << "." << std::endl;
}

// basic error-checking, prints "errorReport" if status failed
bool Loss::check(cudaError_t status, const char* errorReport)
{
    if (status != cudaSuccess)
    {
        fprintf(stderr, errorReport);
        return false;
    }

    return true;
}

// overload to support cudaGetErrorString functionality -- presumes %s call in errorReport
bool Loss::check(cudaError_t status, const char* errorReport, const char* cudaErrorString)
{
    if (status != cudaSuccess)
    {
        fprintf(stderr, errorReport, cudaErrorString);
        return false;
    }

    return true;
}

// overload to support error-code call in errorReport -- presumes %d call in errorReport
bool Loss::check(cudaError_t status, const char* errorReport, cudaError_t errorCode)
{
    if (status != cudaSuccess)
    {
        fprintf(stderr, errorReport, errorCode);
        return false;
    }

    return true;
}


// TEST CODE BELOW -- mostly following CUDA template
//
//// basic vector addition - CUDA error checking
//cudaError_t Loss::addWithCuda(int* c, const int* a, const int* b, unsigned int size)
//{
//    int* dev_a = 0;
//    int* dev_b = 0;
//    int* dev_c = 0;
//    cudaError_t cudaStatus;
//
//    // Allocate GPU buffers for three vectors (two input, one output)
//    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Copy input vectors from host memory to GPU buffers.
//    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    addKernel_call(dev_c, dev_a, dev_b, size);
//
//    // keeping error-checking outside of gpu.cu
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error: // failsafe: on error -> cleanup
//    cudaFree(dev_c);
//    cudaFree(dev_a);
//    cudaFree(dev_b);
//
//    return cudaStatus;
//}
//
//// testing sqrt() functionality - CUDA error checking
//cudaError_t Loss::sqrtWithCuda(const double* arg, double* root, unsigned int size)
//{
//    // define CUDA pointers and error-checking
//    double* dev_arg = 0;
//    double* dev_root = 0;
//    cudaError_t cudaStatus;
//
//    // Allocate GPU buffers for two vector (one input, one output)
//    cudaStatus = cudaMalloc((void**)&dev_arg, size * sizeof(double));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error; // trigger cleanup on failure
//    }
//
//    cudaStatus = cudaMalloc((void**)&dev_root, size * sizeof(double));
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//
//    // Copy input vector from host to device memory: cudaMemcpyHostToDevice flag
//    cudaStatus = cudaMemcpy(dev_arg, arg, size * sizeof(double), cudaMemcpyHostToDevice);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    sqrtKernel_call(dev_arg, dev_root, size);
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "sqrtKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sqrtKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory - cudaMemcpyDeviceToHost flag
//    cudaStatus = cudaMemcpy(root, dev_root, size * sizeof(double), cudaMemcpyDeviceToHost); 
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error: // failsafe: on error -> cleanup
//    cudaFree(dev_arg);
//    cudaFree(dev_root);
//
//    return cudaStatus;
//}

// testing posRoot functionality - first usage of check() error checking
cudaError_t Loss::posRootWithCuda(const void* posRootKernel,
    const cuda::std::complex<double>* arg, cuda::std::complex<double>* root, unsigned int size)
{
    // define CUDA pointers and error-checking
    cuda::std::complex<double>* dev_arg = 0; // is it always efficient to zero out values first?
    cuda::std::complex<double>* dev_root = 0;
    cudaError_t cudaStatus;
    bool working;

    // Allocate GPU buffers for two vector (one input, one output)
    cudaStatus = cudaMalloc((void**)&dev_arg, size * sizeof(cuda::std::complex<double>));
    working = check(cudaStatus, "cudaMalloc dev_arg failed!"); // first cuda-check sets working

    // following cuda-checks only continue if previous ones were successful
    if (working)
    {
    cudaStatus = cudaMalloc((void**)&dev_root, size * sizeof(cuda::std::complex<double>));
    working = check(cudaStatus,"cudaMalloc dev_root failed!");
    }


    // Copy input vector from host to device memory: cudaMemcpyHostToDevice flag
    if (working)
    {
    cudaStatus = cudaMemcpy(dev_arg, arg, size * sizeof(cuda::std::complex<double>), cudaMemcpyHostToDevice);
    working = check(cudaStatus, "cudaMemcpy failed!");
    }

    // Launch a kernel on the GPU with one thread for each element.
    if (working)
    {
        // set up parameters for cudaLaunchKernel
        dim3 gridDim(1, 1, 1);
        dim3 blockDim(size, 1, 1);
        void* kernelArgs[] = { (void*)&dev_arg, (void*)&dev_root };
        int smemSize = 0;
        cudaLaunchKernel(posRootKernel, gridDim, blockDim, kernelArgs, smemSize, NULL);
        cudaStatus = cudaGetLastError(); // check kernel launch
        working = check(cudaStatus, "posRootKernel launch failed: %s\n",
            cudaGetErrorString(cudaStatus)); // query additional kernel error info if failure
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    if (working)
    {
        cudaStatus = cudaDeviceSynchronize();
        working = check(cudaStatus,
            "cudaDeviceSynchronize returned error code %d after launching posRootKernel!\n",
            cudaStatus); // query additional kernel error info if failure
    }

    // Copy output vector from GPU buffer to host memory - cudaMemcpyDeviceToHost flag
    if(working)
    {
        cudaStatus = cudaMemcpy(root, dev_root, size * sizeof(cuda::std::complex<double>),
            cudaMemcpyDeviceToHost);
        working = check(cudaStatus, "cudaMemcpy failed!");
    }

    // failsafe: on error or if all working -> cleanup
    cudaFree(dev_arg);
    cudaFree(dev_root);

    return cudaStatus;
}