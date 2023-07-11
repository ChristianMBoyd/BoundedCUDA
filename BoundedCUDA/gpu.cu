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

    return prefactor * (wVal - posRoot(rootArg));
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

// parallel reduction based on Nvidia sample code
__device__ void sumBlock(cuda::std::complex<double>* sData, const cg::thread_block& block)
{
    const unsigned int tid = block.thread_rank();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    // reduce among thread block tiles
    sData[tid] = cg::reduce(tile32, sData[tid], cg::plus<cuda::std::complex<double>>()); 
    block.sync(); // avoid race condition below

    cuda::std::complex<double> beta = 0; // initialize to zero
    if (block.thread_rank() == 0) 
    {
        // sum over thread block tiles and store result in sData[0]
        for (int i = 0; i < block.size(); i += tile32.size()) 
        {
            beta += sData[i];
        }
        sData[0] = beta;
    }
    block.sync(); // synchronize before return
}

// internal sum over the non-zero Pi0Qn contributions along the diagonal
// calculated per block, and data returned in sData[0]
__device__ void sumPi0Qn(const cg::thread_block& block, cuda::std::complex<double>* sData,
    double q, double w, double delta, double dQ, double Qn, int min, const int tot)
{
    // stride over block before parallel reduction
    for (int i = block.thread_rank(); i < tot; i += block.size())
    {
        // sum over wavevectors Qinput = dQ(i+min)
        sData[block.thread_rank()] += Pi0Qn(q, w, delta, dQ * (i + min), Qn + dQ * (i + min));
    }

    block.sync(); // wait for all threads to fill in entries before moving on

    // per-block reduction, result stored in sData[0]
    sumBlock(sData, block); // includes cg::sync(block) before exit
}

// internal sum over the non-zero Pi0Qnp contributions along the diagonal
// calculated per block, and data returned in sData[0]
__device__ void sumPi0Qnp(const cg::thread_block& block, cuda::std::complex<double>* sData,
    double q, double w, double delta, double dQ, double Qn, double L)
{
    // per-Qn ranges for when Pi0Qnp is non-zero
    const int min = int(cuda::std::ceil(-(Qn + 1) * L / pi));
    const int max = int(cuda::std::floor((1 - Qn) * L / pi));
    const int tot = max - min + 1; // cutoff to include [min,max]

    // stride over block
    for (int i = block.thread_rank(); i < tot; i += block.size())
    {
        // sum over wavevectors Qinput = dQ(i+min)
        sData[block.thread_rank()] += Pi0Qnp(q, w, delta, dQ * (i + min), Qn + dQ * (i + min));
    }

    block.sync(); // wait for all threads to fill in entries before moving on

    // per-block reduction, result stored in sData[0]
    sumBlock(sData, block); // includes cg::sync(block) before exit
}



// kernel definitions below:






// posRoot test of complex math on GPU
__global__ void posRootKernel(const cuda::std::complex<double>* arg, cuda::std::complex<double>* root)
{
    // thread enumeration
    int i = threadIdx.x;

    root[i] = posRoot(arg[i]);
}

// test of cooperative_groups kernel to construct mChi0 -- diagonal only first
// calls for shared memory equal to threadsPerBlock * sizeof(cuda::std::complex<double>)
__global__ void mChi0DiagKernel(const double* QList, cuda::std::complex<double>* mChi0Diag, int size,
    double q, double w, double delta, double L, bool evenPar)
{
    cg::thread_block block = cg::this_thread_block(); // per-block calls
    cg::grid_group grid = cg::this_grid(); // calls across total (blocksPerGrid * threadsPerBlock) threads

    // initialize shared memory to zero for subsequent parallel reductions
    extern cuda::std::complex<double> __shared__ sData[];

    // enumerate sumPi0Qn range
    const int inner = int(cuda::std::floor(L / pi));
    const int innerTot = 2 * inner + 1; // include endpoints of [-inner, inner]
    double dQ = pi / L; // integral index -> wavevector conversion
    
    // stride over blocks to fill in sumPi0Qn values
    for (int i = blockIdx.x; i < size; i += gridDim.x)
    {
        // initialize shared values to zero each loop
        sData[block.thread_rank()] = 0;
        sumPi0Qn(block, sData, q, w, delta, dQ, QList[i], -inner, innerTot); // sum over Pi0Qn
        if (block.thread_rank() == 0)
        {
            mChi0Diag[i] = 0.5 * sData[0]; // result stored in sData[0]
        }
        sData[block.thread_rank()] = 0; // reset sData
        sumPi0Qnp(block, sData, q, w, delta, dQ, QList[i], L); // sum over Pi0Qnp
        if (block.thread_rank() == 0)
        {
            mChi0Diag[i] -= 0.5 * sData[0]; // Pi0Qnp contributes with minus sign
        }
        block.sync();
    }
    grid.sync();

    // edge case
    if (grid.thread_rank() == 0)
    {
        mChi0Diag[0] *= (1 + evenPar);
    }

    // cutoff for when all terms contribute in following loop
    int cutoff = int(cuda::std::ceil(((L / pi) - 1 + evenPar) / 2));
    if (size < cutoff)
    {
        cutoff = size; // out-of-bounds check depending on input (mass) scales
    }

    // stride over grid to fill in remaining entries when all Pi0 terms contribute
    for (int i = grid.thread_rank(); i < cutoff; i += grid.size())
    {
        mChi0Diag[i] -= 0.5 * (Pi0Qn(q, w, delta, QList[i], 0) + Pi0Qn(q, w, delta, 0, QList[i])
            - Pi0Qnp(q, w, delta, QList[i], 0) - Pi0Qnp(q, w, delta, 0, QList[i]));
    }

    grid.sync();

    // stride over grid when only some Pi0 terms contribute
    for (int i = grid.thread_rank() + cutoff; i < size; i += grid.size())
    {
        mChi0Diag[i] -= 0.5 * (Pi0Qn(q, w, delta, 0, QList[i])
            - Pi0Qnp(q, w, delta, QList[i], 0));
    }
}