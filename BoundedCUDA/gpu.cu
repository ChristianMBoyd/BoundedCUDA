#include "gpu.cuh"

// Note: error-checking is kept in .cpp files, below are strictly device code and kernel calls

// device functions below:

// complex square root with positive imaginary part (branch cut along positive reals)
__device__ cuda::std::complex<double> posRoot(cuda::std::complex<double> arg)
{
    return cuda_i * cuda::std::sqrt(-arg);
}

// the Qn-restricted part of Pi0 -- no bounds checking, only pass when |Qn|<1
__device__ cuda::std::complex<double> Pi0Qn(double q, double w, double delta, double Qn, double Qnp)
{
    // reused terms
    const double qq = q * q;
    const double QQ = Qn * Qn;

    // overall scale
    const double prefactor = 1 / (8 * pi * qq);

    // complex expressions
    cuda::std::complex<double> wVal = w + cuda_i * delta - (qq + Qnp * Qnp - QQ);
    cuda::std::complex<double> rootArg = wVal * wVal - 4 * qq * (1 - QQ);

    return prefactor * (wVal - posRoot(rootArg)); // total integral result
}

// the Qnp-restricted part of Pi0 -- no bounds checking, only pass when |Qnp|<1
__device__ cuda::std::complex<double> Pi0Qnp(double q, double w, double delta, double Qn, double Qnp)
{
    // reused terms
    const double qq = q * q;
    const double QQ = Qnp * Qnp;

    // overall scale
    const double prefactor = 1 / (8 * pi * qq);

    // complex expressions
    cuda::std::complex<double> wVal = w + cuda_i * delta - (-qq + QQ - Qn * Qn);
    cuda::std::complex<double> rootArg = wVal * wVal - 4 * qq * (1 - QQ);

    return prefactor * (wVal - posRoot(rootArg)); // total integral result
}







// kernel function and call declarations below:







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

// posRoot test (complex math on GPU)
__global__ void posRootKernel(const cuda::std::complex<double>* arg, cuda::std::complex<double>* root)
{
    // thread enumeration
    int i = threadIdx.x;

    root[i] = posRoot(arg[i]);
}