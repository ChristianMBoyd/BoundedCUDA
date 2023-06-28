#include "gpu.cuh"
// Note: error-checking is kept in .cpp files, below are strictly device code and kernel calls
// 
//device constants
__constant__ cuda::std::complex<double> cuda_i(0, 1);
__constant__ double pi = 3.141592653589793238463;



// device functions below:



// complex square root with positive imaginary part (branch cut along positive real axis)
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





// kernel definitions below:






// posRoot test (complex math on GPU)
__global__ void posRootKernel(const cuda::std::complex<double>* arg, cuda::std::complex<double>* root)
{
    // thread enumeration
    int i = threadIdx.x;

    root[i] = posRoot(arg[i]);
}