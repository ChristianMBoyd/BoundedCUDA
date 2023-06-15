#include "gpu.cuh"

// Note: error-checking is kept in .cpp files, below are strictly device code and kernel calls

// kernell call to nvidia test code
void addKernel_call(int* c, const int* a, const int* b,unsigned int size)
{
    addKernel << < 1, size >> > (c, a, b);// this arrangement works, previous versions of <<<#,#>>> flagged at compile time
}

// nvidia test code
__global__ void addKernel(int* c, const int* a, const int* b)
{
    // thread enumeration
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// kernel call to sqrt test
void sqrtKernel_call(const double* arg, double* root, unsigned int size)
{
    sqrtKernel << < 1, size >> > (arg, root);
}

// sqrt test
__global__ void sqrtKernel(const double* arg, double* root)
{
    // thread enumeration
    int i = threadIdx.x;
    root[i] = cuda::std::sqrt(arg[i]);
}

// kernel call to posRoot test
void posRootKernel_call(const cuda::std::complex<double>* arg, cuda::std::complex<double>* root, unsigned int size)
{
    posRootKernel << < 1, size >> > (arg, root);
}

// posRoot test on complex numbers
__global__ void posRootKernel(const cuda::std::complex<double>* arg, cuda::std::complex<double>* root)
{
    // thread enumeration
    int i = threadIdx.x;
    // "i" def 
    const cuda::std::complex<double> im(0, 1.);

    root[i] = im * cuda::std::sqrt(-arg[i]);
}