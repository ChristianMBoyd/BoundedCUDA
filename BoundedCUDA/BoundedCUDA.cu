#include "gpu.cuh"
#include "Loss.h"

int main()
{
   // To do:
   //  1) Fix numerical error with mChi0Diag
   //       do more tests to ensure calculating the correct objects
   //  2) Implement parallelized matrix/vector initializations
   //      i.e., implement in a way that allows for direct cuBLAS calls after
   //  3) Implement cuBLAS methods for subsequent vector and matrix operations
   //  4) Implement cuSOLVER for the linear solve step

    Loss g; // initialize GPU and CUDA methods
    g.deviceQuery(); // print out basic GPU facts
    // error-checking
    cudaError_t cudaStatus;
    bool working = true;

    //// posRoot test for complex values, done on device
    //const int arraySize = 4; // "unsigned int" from cpp POV
    //const cuda::std::complex<double> arg[arraySize] = { cuda::std::complex<double>(1,1),
    //cuda::std::complex<double>(-1,1) ,cuda::std::complex<double>(1,-1) ,cuda::std::complex<double>(-1,-1) };
    //cuda::std::complex<double> root[arraySize] = { 0 };
    // 
    //cudaStatus = g.posRootWithCuda(posRootKernel, arg, root, arraySize);
    //g.check(working, cudaStatus, "posRootWithCuda() failed to complete!");
    //if (!working)
    //{
    //    return 1; // main() error
    //}

    //// printout results
    //std::cout << "The result of sqrt({";
    //for (int i = 0; i < arraySize; i++)
    //{
    //    std::cout << "(" << arg[i].real() << "," << arg[i].imag() << "), ";
    //}
    //std::cout << "}):\n";
    //std::cout << "{";
    //for (int i = 0; i < arraySize; i++)
    //{
    //    std::cout << "(" << root[i].real() << "," << root[i].imag() << "), ";
    //}
    //std::cout << "})" << std::endl;
    //std::cout << " " << std::endl; // blank line before further output

    // mChi0Diag test of kernel launching

    // input parameters
    const double q = 0.1, w = 1.1, delta = 0.1, L = 100, cutoff = 5;
    const int nMax = g.nMax(cutoff, L);
    const bool evenMax = g.evenQ(nMax);
    const int par = 1;
    const bool evenPar = g.evenQ(par);
    const int size = g.totNum(nMax, evenPar);
    std::cout << "Size of mChi0Diag and QList: " << size << "." << std::endl;

    // initialize host memory
    double* QList = new double[size];
    g.initializeQList(QList, L, size, evenPar); // fill in QList
    cuda::std::complex<double>* mChi0Diag = new cuda::std::complex<double>[size];

    // calculate diagonal entries of mChi0 on GPU, returned in mChi0Diag
    cudaStatus = g.mChi0DiagLaunch(mChi0DiagKernel, QList, mChi0Diag, size, q, w, delta, L, evenPar);
    g.check(working, cudaStatus, "mChi0DiagLaunch() failed!");
    
    if (working)
    {
        cuda::std::complex<double> sum = 0;
        for (int i = 0; i < size; i++)
        {
            sum += mChi0Diag[i];
        }

        std::cout << "Sum of mChi0Diag: (" << sum.real() << "," << sum.imag() << ")" << std::endl;
        std::cout << "" << std::endl;
    }

    // array cleanup
    delete QList;
    delete mChi0Diag;

    // END OF mChi0Diag TEST

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    g.check(working, cudaStatus, "cudaDeviceReset failed!");

    // closing input to have window hang after printing results
    char input;
    std::cout << "\nEnter any input to close.\n";
    std::cin >> input;

    return 0;
}
