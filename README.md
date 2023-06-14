# BoundedCUDA
Surface loss function of a bounded electron gas calculated on GPU with CUDA.

My previous implementation in a C++ environment was based on Eigen + MKL BLAS subroutines to handle the linear algebra on multi-threaded CPUs.  This project will utilize the GPU structure to take advantage of parallelizability in the matrix initializations.
