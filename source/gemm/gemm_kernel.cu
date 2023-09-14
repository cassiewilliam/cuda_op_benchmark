#include "gemm_runner.hpp"
#include "cutlass/uint128.h"

#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;
using uint128_t = cutlass::uint128_t;

// 1. HGEMM Naive Kernel
__global__ void hgemm_naive(bool A_transpose, bool B_transpose, int m, int n, int k, float alpha, const half *A, const half *B, float beta, const float *C, float *D)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < m && y < n)
    {
        float sum = 0.0;

        for (int i = 0; i < k; ++i)
        {
            sum += (float(A[x * k + i]) * float(B[y * k + i]));
        }

        // NOTE: D = α * (A @ B) + β * C
        D[x * n + y] = alpha * sum + beta * C[x * n + y];
    }
}

// 2. HGEMM with Tensor Core wmma API
__global__ void wmma_m16n16k16_kernel(bool  A_transpose,
                                      bool  B_transpose,
                                      int   m,
                                      int   n,
                                      int   k,
                                      float alpha,
                                      const half *__restrict__ A,
                                      const half *__restrict__ B,
                                      float beta,
                                      const float *__restrict__ C,
                                      float *__restrict__ D)
{
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    const int lda = k;
    const int ldb = k;
    const int ldc = n;

    // NOTE: warp对应矩阵M方向实际的index位置
    const int warp_m_index = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    // NOTE: warp对应矩阵N方向实际的index位置
    const int warp_n_index = (blockIdx.y * blockDim.y + threadIdx.y);

    // NOTE: 声明WMMA所需要的寄存器
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>              acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>              cd_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // NOTE: k是内循环
    for (int i = 0; i < k; i += WMMA_K)
    {
        int a_block_row = warp_m_index * WMMA_M;
        int a_block_col = i;

        int b_block_row = warp_n_index * WMMA_N; // NOTE: 列主序的矩阵
        int b_block_col = i;

        if ((a_block_row < m && a_block_col < k) && (b_block_row < n && b_block_col < k))
        {
            wmma::load_matrix_sync(a_frag, A + a_block_row * lda + a_block_col, lda);
            wmma::load_matrix_sync(b_frag, B + b_block_row * ldb + b_block_col, ldb);

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    size_t c_block_row = warp_m_index * WMMA_M;
    size_t c_block_col = warp_n_index * WMMA_N;

    // NOTE: 写出结果
    if (c_block_row < m && c_block_col < n)
    {
        wmma::load_matrix_sync(cd_frag, C + c_block_row * ldc + c_block_col, ldc, wmma::mem_row_major);

#pragma unroll
        for (int i = 0; i < cd_frag.num_elements; i++)
        {
            cd_frag.x[i] = alpha * acc_frag.x[i] + beta * cd_frag.x[i];
        }

        wmma::store_matrix_sync(D + c_block_row * ldc + c_block_col, cd_frag, ldc, wmma::mem_row_major);
    }
}

// reference from https://github.com/Bruce-Lee-LY/cuda_hgemm/blob/master/src/common/ptx.h
#define LDMATRIX_X2(R0, R1, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1)                                         \
                 : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
                 : "r"(addr))

#define HMMA16816(RD0, RD1, RD2, RD3, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1, RC2, RC3)                                                    \
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n" \
                 : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3)                                                                              \
                 : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1), "r"(RC2), "r"(RC3))

// 3. HGEMM with Tensor Core wmma PTX API
__global__ void mma_m16n8k16_kernel(bool  A_transpose,
                                    bool  B_transpose,
                                    int   m,
                                    int   n,
                                    int   k,
                                    float alpha,
                                    const half *__restrict__ A,
                                    const half *__restrict__ B,
                                    float beta,
                                    const float *__restrict__ C,
                                    float *__restrict__ D)
{
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int WARP_SIZE = 32;

    const int lda = k;
    const int ldb = k;
    const int ldc = n;

    const int warp_row_index = blockIdx.x;
    const int warp_col_index = blockIdx.y;

    // NOTE: warp对应矩阵M方向实际的index位置
    const int warp_m_offset = warp_row_index * MMA_M;
    // NOTE: warp对应矩阵N方向实际的index位置
    const int warp_n_offset = warp_col_index * MMA_N;

    if (warp_m_offset >= m || warp_n_offset >= n)
    {
        return;
    }

    __shared__ half  A_shmem[MMA_M][MMA_K];
    __shared__ half  B_shmem[MMA_N][MMA_K];
    __shared__ float C_shmem[MMA_M][MMA_N];
    __shared__ float D_shmem[MMA_M][MMA_N];

    const size_t lane_id = threadIdx.x % WARP_SIZE;

    uint32_t RAcc[4] = {0, 0, 0, 0};
    float    RC[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (size_t i = 0; i < k; i += MMA_K)
    {
        // Step1: A and B Global To Shared Memory
        *((uint128_t *)(&A_shmem[lane_id / 2][(lane_id % 2) * 8])) =
            *((uint128_t *)(A + (warp_m_offset + lane_id / 2) * lda + i + (lane_id % 2) * 8));

        if (lane_id < MMA_N * 2)
        {
            *((uint128_t *)(&B_shmem[lane_id / 2][(lane_id % 2) * 8])) =
                *((uint128_t *)(B + (warp_n_offset + lane_id / 2) * ldb + i + (lane_id % 2) * 8));
        }

        __syncthreads();

        uint32_t RA[4];
        uint32_t RB[2];

        // Step2: A and B Global To Shared Memory with ldmatrix
        uint32_t A_shmem_lane_addr = __cvta_generic_to_shared(&A_shmem[lane_id % 16][(lane_id / 16) * 8]);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], A_shmem_lane_addr);

        uint32_t B_shmem_lane_addr = __cvta_generic_to_shared(&B_shmem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LDMATRIX_X2(RB[0], RB[1], B_shmem_lane_addr);

        // Step3: TensorCore计算矩阵乘法
        HMMA16816(RAcc[0], RAcc[1], RAcc[2], RAcc[3], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RAcc[0], RAcc[1], RAcc[2], RAcc[3]);

        __syncthreads();
    }

    // Step4: 加载C Global => Shared => Register 并且进行计算
    *((uint128_t *)(&C_shmem[lane_id / 2][0])) = *((uint128_t *)(&C[(warp_m_offset + lane_id / 2) * ldc + warp_n_offset + 0]));
    *((uint128_t *)(&C_shmem[lane_id / 2][4])) = *((uint128_t *)(&C[(warp_m_offset + lane_id / 2) * ldc + warp_n_offset + 4]));

    // Shared To Register
    RC[0] = C_shmem[lane_id / 4 + 0][(lane_id % 4) * 2 + 0];
    RC[1] = C_shmem[lane_id / 4 + 0][(lane_id % 4) * 2 + 1];
    RC[2] = C_shmem[lane_id / 4 + 8][(lane_id % 4) * 2 + 0];
    RC[3] = C_shmem[lane_id / 4 + 8][(lane_id % 4) * 2 + 1];

    // NOTE: D = α * (A @ B) + β * C
    for (int i = 0; i < 4; ++i)
    {
        RC[i] = alpha * ((float *)RAcc)[i] + beta * RC[i];
    }

    // Step5: 输出Register File To Shared Memory
    D_shmem[lane_id / 4 + 0][(lane_id % 4) * 2 + 0] = RC[0];
    D_shmem[lane_id / 4 + 0][(lane_id % 4) * 2 + 1] = RC[1];
    D_shmem[lane_id / 4 + 8][(lane_id % 4) * 2 + 0] = RC[2];
    D_shmem[lane_id / 4 + 8][(lane_id % 4) * 2 + 1] = RC[3];

    __syncthreads();

    // Step6. 结果写出，每次写出128bit 4个float数据，一行分两次写出
    *((uint128_t *)(&D[(warp_m_offset + lane_id / 2) * ldc + warp_n_offset + 0])) = *((uint128_t *)(&D_shmem[lane_id / 2][0]));
    *((uint128_t *)(&D[(warp_m_offset + lane_id / 2) * ldc + warp_n_offset + 4])) = *((uint128_t *)(&D_shmem[lane_id / 2][4]));
}

// 4. HGEMM with Tensor Core wmma PTX API, Async Load Memory Global to Shared and Double Buffer
__global__ void mma_m16n8k16_V2_kernel(bool  A_transpose,
                                       bool  B_transpose,
                                       int   m,
                                       int   n,
                                       int   k,
                                       float alpha,
                                       const half *__restrict__ A,
                                       const half *__restrict__ B,
                                       float beta,
                                       const float *__restrict__ C,
                                       float *__restrict__ D)
{
    // TODO: 暂未实现
}

// 5. HGEMM with Cutlass API
__global__ void cutlass_m16n8k16_kernel(bool  A_transpose,
                                        bool  B_transpose,
                                        int   m,
                                        int   n,
                                        int   k,
                                        float alpha,
                                        const half* __restrict__ A,
                                        const half* __restrict__ B,
                                        float beta,
                                        half* __restrict__ C)
{
    // TODO: 暂未实现
}

void gemm_runner(bool         transa,
                 bool         transb,
                 int          m,
                 int          n,
                 int          k,
                 const float *alpha,
                 const void * A,
                 DataType     Atype,
                 const void * B,
                 DataType     Btype,
                 const float *beta,
                 const void * C,
                 DataType     Ctype,
                 void *       D,
                 DataType     Dtype,
                 KernelType   algoType)
{
    if (Atype == DataType::FLOAT16 && Btype == DataType::FLOAT16 && Ctype == DataType::FLOAT32 && Dtype == DataType::FLOAT32)
    {
        if (algoType == KernelType::KT_HGEMM_Naive)
        {
            // NOTE: 直接将结果看起一个二维的平面，然后以32x32的块进行切分
            dim3 grid_dim(UP_DIV(m, 32), UP_DIV(n, 32), 1);
            dim3 block_dim(32, 32, 1);

            hgemm_naive<<<grid_dim, block_dim>>>(false, false, m, n, k, *alpha, (const half *)A, (const half *)B, *beta, (const float *)C, (float *)D);
        }
        else if (algoType == KernelType::KT_HGEMM_WMMA)
        {
            constexpr int WMMA_M = 16;
            constexpr int WMMA_N = 16;
            constexpr int WMMA_K = 16;
            constexpr int WARP_SIZE = 32;

            dim3 block_dim(WARP_SIZE);
            dim3 grid_dim(UP_DIV(m, (WMMA_M * block_dim.x / WARP_SIZE)), UP_DIV(n, (WMMA_N * block_dim.y)));

            wmma_m16n16k16_kernel<<<grid_dim, block_dim>>>(false, false, m, n, k, *alpha, (const half *)A, (const half *)B, *beta, (const float *)C, (float *)D);
        }
        else if (algoType == KernelType::KT_HGEMM_MMA_PTX)
        {
            constexpr int MMA_M = 16;
            constexpr int MMA_N = 8;
            constexpr int MMA_K = 16;
            constexpr int WARP_SIZE = 32;

            dim3 block_dim(WARP_SIZE);
            dim3 grid_dim(UP_DIV(m, MMA_M), UP_DIV(n, MMA_N));

            mma_m16n8k16_kernel<<<grid_dim, block_dim>>>(false, false, m, n, k, *alpha, (const half *)A, (const half *)B, *beta, (const float *)C, (float *)D);
        }
        else if (algoType == KernelType::KT_HGEMM_MMA_PTX_OPT)
        {
            
        }
        else if (algoType == KernelType::KT_HGEMM_CUTLASS)
        {

        }
    }
    else
    {
        LOGE("only support datatype with half");
    }
}
