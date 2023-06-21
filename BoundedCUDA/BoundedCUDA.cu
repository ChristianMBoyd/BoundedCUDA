
#include "Loss.h"

int main()
{
    const int arraySize = 4; // "unsigned int" from cpp POV
    const cuda::std::complex<double> arg[arraySize] = { cuda::std::complex<double>(1,1),
    cuda::std::complex<double>(-1,1) ,cuda::std::complex<double>(1,-1) ,cuda::std::complex<double>(-1,-1) };
    cuda::std::complex<double> root[arraySize] = { 0 };

    Loss g; // initialize GPU and CUDA methods
    g.deviceQuery(); // print out basic GPU facts

    // To do:
    //  1) Implement mathematical functions -- last did Pi0Qn() and Pi0Qnp()
    //  2) Implement parallelized matrix/vector initializations
    //      i.e., implement in a way that allows for direct cuBLAS calls after
    //  3) Implement cuBLAS methods for subsequent vector and matrix operations
    //  4) Implement cuSOLVER for the linear solve step

    // current test -- posRoot on complex values, done on device
    cudaError_t cudaStatus = g.posRootWithCuda(arg, root, arraySize);
    bool working = g.check(cudaStatus,"posRootWithCuda() failed to complete!");
    if (!working)
    {
        return 1; // main() error
    }

    // printout results
    std::cout << "The result of sqrt({";
    for (int i = 0; i < arraySize; i++)
    {
        std::cout << "(" << arg[i].real() << "," << arg[i].imag() << "), ";
    }
    std::cout << "}):\n";
    std::cout << "{";
    for (int i = 0; i < arraySize; i++)
    {
        std::cout << "(" << root[i].real() << "," << root[i].imag() << "), ";
    }
    std::cout << "})" << std::endl;

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    working = g.check(cudaStatus, "cudaDeviceReset failed!");
    if (!working)
    {
        return 1; // main() error
    }

    // closing preamble to have window hang after printing results
    char input;
    std::cout << "\nEnter any input to close.\n";
    std::cin >> input;

    return 0;
}
