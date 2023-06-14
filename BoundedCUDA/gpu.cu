#include "gpu.cuh"

// kernell call to nvidia test code
void addKernel_call(unsigned int size, int* c, const int* a, const int* b)
{
    addKernel <<< 1, size >>> (c, a, b);// this arrangement works, previous versions of <<<#,#>>> flagged at compile time
}

// nvidia test code
__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
