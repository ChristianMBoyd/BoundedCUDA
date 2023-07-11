#include "Loss.h"

Loss::Loss()
{
    cudaError_t cudaStatus = initializeDevice(); // already checked in initializeDevice(), may not need
}

// assumes single GPU at device "0," extracts useful info into deviceProp and sets device for subsequent use
cudaError_t Loss::initializeDevice()
{
    // error checking
    cudaError_t cudaStatus;
    bool working = true;

    // extract basic information from device
    cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
    MAX_BLOCKS_PER_MP = deviceProp.maxBlocksPerMultiProcessor;
    MAX_THREADS_PER_BLOCK = deviceProp.maxThreadsPerBlock;
    NUM_MP = deviceProp.multiProcessorCount;
    MAX_THREADS_PER_MP = deviceProp.maxThreadsPerMultiProcessor;
    MAX_SMEM_PER_BLOCK = deviceProp.sharedMemPerBlock;
    MAX_SMEM_PER_MP = deviceProp.sharedMemPerMultiprocessor;

    check(working, cudaStatus, "cudaGetDeviceProperties failed! Check GPU configuration.");
    
    if (working)
    {
        // set device for CUDA use
        cudaStatus = cudaSetDevice(0);
        check(working, cudaStatus, "cudaSetDevice() failed on device 0! Check GPU configuration.");
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
    std::cout << "Max shared memory per multiprocessor: " << MAX_SMEM_PER_MP <<
        " bytes." << std::endl;
    std::cout << "Max threads per block: " << MAX_THREADS_PER_BLOCK << "." << std::endl;
    std::cout << "Max shared memory per block: " << MAX_SMEM_PER_BLOCK <<
        " bytes." << std::endl;
    std::cout << "'1' if concurrent kernels: " << deviceProp.concurrentKernels
        << "." << std::endl;
    std::cout << "[output]>0 if async host-device memory transfer + kernel execution: "
        << deviceProp.asyncEngineCount << "." << std::endl;
    std::cout << "'1' if cooperative launch support: " << deviceProp.cooperativeLaunch
        << "." << std::endl;
    std::cout << " " << std::endl; // blank line before next output somewhere else
}

// basic error-checking: presumes "working" initialized to true and prints "errorReport" on cuda error
void Loss::check(bool& working, cudaError_t status, const char* errorReport)
{
    if (status != cudaSuccess)
    {
        const char* errorName = cudaGetErrorName(status);
        const char* errorDesc = cudaGetErrorString(status);
        fprintf(stderr, "%s\nError: %s\nDescription: %s\n", errorReport, errorName, errorDesc);
        // signal to halt code progress
        working = false;
    }
}

// cuda calls for optimal occupany checked against device properties, smemSize is per-block shared memory requested
// threadSmem is the per-thread shared memory requested, used to calculate smemSize
cudaError_t Loss::optimizeLaunchParameters(int& threadsPerBlock, int& blocksPerGrid, int&smemSize,
    const int threadSmem, const void* kernelFunc)
{
    bool working = true;
    cudaError_t cudaStatus = cudaOccupancyMaxPotentialBlockSize(&blocksPerGrid, &threadsPerBlock, kernelFunc);
    check(working, cudaStatus, "cudaOccupancyMaxPotentialBlockSize failed! Check thread and block initialization.");

    if (working)
    {
        // determine per-block shared memory based on per-thread memory requested
        smemSize = threadsPerBlock * threadSmem;

        // check results against device constraints (not currently checking smemSize)
        threadsPerBlock = std::min(threadsPerBlock, MAX_THREADS_PER_BLOCK);
        int maxBlocks = NUM_MP * MAX_THREADS_PER_MP / MAX_THREADS_PER_BLOCK;
        blocksPerGrid = std::min(blocksPerGrid, maxBlocks);

        int blocksPerMP = 0;
        cudaStatus = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerMP, kernelFunc, threadsPerBlock, smemSize);
        check(working, cudaStatus, "cudaOccupancyMaxActiveBlocksPerMultiprocessor failed! Check thread and block initialization.");
        blocksPerGrid = NUM_MP * blocksPerMP;
    }

    if (working)
    {
        // check against device parameters
        blocksPerGrid = std::min(NUM_MP * MAX_BLOCKS_PER_MP, blocksPerGrid);
    }

    return cudaStatus;
}

// checks if an integer is even or odd
bool Loss::evenQ(const int val)
{
    return (val % 2 == 0);
}

// enumerate maximum (absolute value) wavevector integral index
int Loss::nMax(const double cutoff, const double L)
{
    return int(std::ceil(cutoff * L / pi));
}

// enumerate total number of non-negative wavevector entries
int Loss::totNum(const int nMax, const bool evenPar)
{
    bool evenMax = evenQ(nMax);
    return int(std::floor((nMax + 1) / 2)) + int(std::floor((evenMax + evenPar) / 2));
}

// fill in the non-negative entries of QList
void Loss::initializeQList(double QList[], const double L, const int tot, const bool evenPar)
{
    const double dQ = pi / L;

    for (int counter = 0; counter < tot; counter++)
    {
        QList[counter] = dQ * (1 - evenPar + 2 * counter);
    }
}








// kernel calls below








// kernel to fill in the diagonal, symmetry-reduced entries in mChi0Diag
cudaError_t Loss::mChi0DiagLaunch(const void* mChi0DiagKernel, const double QList[], cuda::std::complex<double> mChi0Diag[], int size,
    double q, double w, double delta, double L, bool evenPar)
{
    // initialize device pointers 
    double* dev_QList = 0;
    cuda::std::complex<double>* dev_mChi0Diag = 0;
    
    // initialize error-checking
    cudaError_t cudaStatus;
    bool working = true;

    // Allocate GPU buffers
    cudaStatus = cudaMalloc((void**)&dev_QList, size * sizeof(double));
    check(working, cudaStatus, "cudaMalloc dev_QList failed!");

    if (working)
    {
        cudaStatus = cudaMalloc((void**)&dev_mChi0Diag, size * sizeof(cuda::std::complex<double>));
        check(working, cudaStatus, "cudaMalloc dev_mChi0Diag failed!");
    }

    // copy QList into device memory: cudaMemcpyHostToDevice flag
    if (working)
    {
        cudaStatus = cudaMemcpy(dev_QList, QList, size * sizeof(double), cudaMemcpyHostToDevice);
        check(working, cudaStatus, "cudaMemcpy on QList failed!");
    }

    // determine optimal launch parameters (likely implement custom variant later on)
    int threadsPerBlock = 0, blocksPerGrid = 0, smemSize = 0;
    // threadSmem is the per-thread shared memory requested
    // smemSize is the per-block shared memory, determined using theadSmem
    int threadSmem = sizeof(cuda::std::complex<double>);
    if (working)
    {
        cudaStatus = optimizeLaunchParameters(threadsPerBlock, blocksPerGrid, smemSize,
            threadSmem, mChi0DiagKernel);
        check(working, cudaStatus, "optimizeLaunchParameters for mChi0DiagKernel failed!");
        // cuda thread and block parameters
    }
    dim3 blockDim(threadsPerBlock, 1, 1);
    dim3 gridDim(blocksPerGrid, 1, 1);

    // launch mChi0DiagKernel
    if (working)
    { 
        std::cout << "Launching mChi0DiagKernel with " << blocksPerGrid << " blocks of "
            << threadsPerBlock << " threads...";
        void* kernelArgs[] = { (void*)&dev_QList, (void*)&dev_mChi0Diag, (void*)&size,
            (void*)&q, (void*)&w, (void*)&delta, (void*)&L, (void*)&evenPar };
        cudaLaunchCooperativeKernel(mChi0DiagKernel, gridDim, blockDim, kernelArgs, smemSize, NULL);
        cudaStatus = cudaGetLastError(); // check kernel launch
        check(working, cudaStatus, "cudaLaunchCooperativeKernel failed on mChi0DiagKernel!"); // query additional kernel error info
    }

    // wait for kernel to finish and collect errors afterward
    if (working)
    {
        cudaStatus = cudaDeviceSynchronize();
        std::cout << "Finished." << std::endl;
        check(working, cudaStatus,
            "cudaDeviceSynchronize failed after launching mChi0DiagKernel!"); // query additional kernel error info 
    }

    // copy result to host memory: cudamemcpyDeviceToHost flag
    if (working)
    {
        cudaStatus = cudaMemcpy(mChi0Diag, dev_mChi0Diag, size * sizeof(cuda::std::complex<double>),
                cudaMemcpyDeviceToHost);
        check(working, cudaStatus, "cudaMemcpy failed to copy mChi0Diag to host!");
    }

    // failsafe clean-up: free device memory on failure or success
    cudaFree(dev_QList);
    cudaFree(dev_mChi0Diag);

    return cudaStatus;
}

// testing posRoot functionality - first usage of check() error checking
// CAUTION: likely out-dated/deprecated usage 
cudaError_t Loss::posRootWithCuda(const void* posRootKernel,
    const cuda::std::complex<double>* arg, cuda::std::complex<double>* root, unsigned int size)
{
    // define CUDA pointers and error-checking
    cuda::std::complex<double>* dev_arg = 0;
    cuda::std::complex<double>* dev_root = 0;
    cudaError_t cudaStatus;
    bool working = true; // initialize

    // Allocate GPU buffers for two vectors (one input, one output)
    cudaStatus = cudaMalloc((void**)&dev_arg, size * sizeof(cuda::std::complex<double>));
    check(working, cudaStatus, "cudaMalloc dev_arg failed!"); // error-checking cuda calls

    // following cuda-checks only continue if previous ones were successful
    if (working)
    {
        cudaStatus = cudaMalloc((void**)&dev_root, size * sizeof(cuda::std::complex<double>));
        check(working, cudaStatus, "cudaMalloc dev_root failed!");
    }


    // Copy input vector from host to device memory: cudaMemcpyHostToDevice flag
    if (working)
    {
        cudaStatus = cudaMemcpy(dev_arg, arg, size * sizeof(cuda::std::complex<double>), cudaMemcpyHostToDevice);
        check(working, cudaStatus, "cudaMemcpy failed!");
    }

    // Launch a kernel on the GPU with one thread for each element.
    if (working)
    {
        // set up parameters for cudaLaunchKernel
        dim3 blocksPerGrid(1, 1, 1);
        dim3 threadsPerBlock(size, 1, 1);
        void* kernelArgs[] = { (void*)&dev_arg, (void*)&dev_root };
        int smemSize = 0;
        cudaLaunchKernel(posRootKernel, blocksPerGrid, threadsPerBlock, kernelArgs, smemSize, NULL);
        cudaStatus = cudaGetLastError(); // check kernel launch
        check(working, cudaStatus, "posRootKernel launch failed!"); 
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // errors encountered during the launch.
    if (working)
    {
        cudaStatus = cudaDeviceSynchronize();
        check(working, cudaStatus,
            "cudaDeviceSynchronize returned error code %d after launching posRootKernel!");
    }

    // Copy output vector from GPU buffer to host memory - cudaMemcpyDeviceToHost flag
    if (working)
    {
        cudaStatus = cudaMemcpy(root, dev_root, size * sizeof(cuda::std::complex<double>),
            cudaMemcpyDeviceToHost);
        check(working, cudaStatus, "cudaMemcpy failed!");
    }

    // failsafe: on error or if all working -> cleanup
    cudaFree(dev_arg);
    cudaFree(dev_root);

    return cudaStatus;
}
