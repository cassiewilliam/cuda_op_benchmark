#include "cuda_runtime.hpp"

// NOTE: reference from cutlass examples
template<typename T>
__global__ void initialize_matrix_kernel(T* matrix, int rows, int columns, int seed)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < rows && j < columns)
    {
        int offset = i + j * rows;

        // Generate arbitrary elements.
        int const k     = 16807;
        int const m     = 16;
        T         value = T(((offset + seed) * k % m) - m / 2);
        matrix[offset]  = value;
    }
}

/// Simple function to initialize a matrix to arbitrary small integers.
template<typename T>
cudaError_t initialize_matrix(T* matrix, int rows, int columns, int seed)
{
    dim3 block(16, 16);
    // NOTE: UP_DIV(), 保证线程数足够，每个block 16*16个线程
    dim3 grid((rows + block.x - 1) / block.x, (columns + block.y - 1) / block.y);

    initialize_matrix_kernel<T><<<grid, block>>>(matrix, rows, columns, seed);

    return cudaGetLastError();
}

// Specialize template function
template
cudaError_t initialize_matrix<float>(float* matrix, int rows, int columns, int seed);
template
cudaError_t initialize_matrix<half>(half* matrix, int rows, int columns, int seed);
template
cudaError_t initialize_matrix<int8_t>(int8_t* matrix, int rows, int columns, int seed);
