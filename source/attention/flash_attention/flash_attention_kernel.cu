#include "flash_attention_kernel.cuh"
#include "cutlass/uint128.h"

#include <stdint.h>

using uint128_t = cutlass::uint128_t;

// NOTE：输入QKV的数据排布 batch0:
//                             seq_len0:
//                                      head0:
//                                            elem0, elem1, ..., elemt
//                                      head1:
//                                            elem0, elem1, ..., elemt
//                                      ......
//                                      headn:
//                                            elem0, elem1, ..., elemt
//                             seq_len1:
//                                      head0:
//                                            elem0, elem1, ..., elemt
//                                      head1:
//                                            elem0, elem1, ..., elemt
//                                      ......
//                                      headn:
//                                            elem0, elem1, ..., elemt
//                             ...... ......
//                             seq_lenx:
//                                      head0:
//                                            elem0, elem1, ..., elemt
//                                      head1:
//                                            elem0, elem1, ..., elemt
//                                      ......
//                                      headn:
//                                            elem0, elem1, ..., elemt
__device__ 
void flash_attention_1block_kernel(const half*  q, 
                                   const half*  k, 
                                   const half*  v, 
                                   half*        o,
                                   const int    batch_size,
                                   const int    seq_len,
                                   const int    num_heads,
                                   const int    head_dim,
                                   const int    batch_index,
                                   const int    heads_index,
                                   const int    block_index)
{
    const int thread_idx = threadIdx.x;

    const int warp_id = thread_idx >> 5;
    const int lane_id = thread_idx & 0x1f;

    constexpr int kBlockM = 128;
    constexpr int kBlockN = 64;
    constexpr int kWarps = 4;

    // NOTE: current only support head_dim is 128
    constexpr int kBlockK = 128;
    const int kBlockThreds = kWarps * 32;

    // NOTE: 线程块超出真实数据范围
    if (block_index * kBlockM >= seq_len) return;

    int n_block_max = UP_DIV(seq_len, kBlockN);

    // NOTE: 取的是min, m_block = [0, 1, ..., t],
    // n_block_max会根据mask对角矩阵之计算需要计算的位置
    n_block_max = std::min(n_block_max, UP_DIV((block_index + 1) * kBlockM, kBlockN));

    const int batch_stride = seq_len * num_heads * head_dim;
    const int seq_stride = num_heads * head_dim;
    const int head_stride = head_dim;

    const int row_offset_q = batch_index * batch_stride +
                             block_index * kBlockM * seq_stride +
                             heads_index * head_stride;

    const int row_offset_k = batch_index * batch_stride + 
                             (n_block_max - 1) * kBlockN * seq_stride +
                             heads_index * head_stride;

    const int row_offset_v = batch_index * batch_stride + 
                             (n_block_max - 1) * kBlockN * seq_stride +
                             heads_index * head_stride;

    auto gmem_q_ptr = q + row_offset_q;
    auto gmem_k_ptr = k + row_offset_k;
    auto gmem_v_ptr = v + row_offset_v;

    // NOTE: 128 * 64 * sizeof(half) = 16kb * 3 = 48kb
    //       M128N64k128，每个线程都去(M * K) / (kWarps * 32) = 128个q数据
    __shared__ half smem_q_ptr[kBlockM][kBlockK];
    __shared__ half smem_k_ptr[kBlockK][kBlockN];
    __shared__ half smem_v_ptr[kBlockK][kBlockN];

    __shared__ half smem_half_o[kBlockM][kBlockN];


    // Step1: Load Global Memory To Shared Memory
    /// Step1.1: 异步加载数据Q, 每次拷贝128bit, 8个数据
    constexpr int load_elem_size = sizeof(uint128_t);

    // NOTE: 每个线程加载的数据需要考虑Stride，比如一行数据为128，但是加载数据为192，
    //       这个时候加载数据会跨行，计算offset比较麻烦
    //       在当前例子中，每个线程加载数据刚好等于一行数据大小
    constexpr int q_elem_num_per_thread = (kBlockM * kBlockK) / kBlockThreds;
    const int q_load_times_per_thread = q_elem_num_per_thread / load_elem_size;

    // NOTE: 每个线程加载一行数据，每次加载8个，需要加载16次
    for (int i = 0; i < q_load_times_per_thread; ++i)
    {
        // uint32_t* q_gmem_head_ptr = (uint32_t*)(gmem_q_ptr + thread_idx * seq_stride + i * load_elem_size);
        // uint32_t* q_smem_head_ptr = (uint32_t*)(&smem_q_ptr[thread_idx][i * load_elem_size]);

        // asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
        //              :: "r"(q_gmem_head_ptr),
        //                 "l"(q_smem_head_ptr),
        //                 "n"(load_elem_size));
    }

    asm volatile("cp.async.commit_group;\n" ::);

    /// Step1.2: 异步加载数据K, 每次拷贝128bit, 16个数据
    constexpr int k_elem_num_per_thread = (kBlockN * kBlockK) / kBlockThreds;
    const int k_load_times_per_thread = k_elem_num_per_thread / load_elem_size;

    int n_block = n_block_max - 1;

    // NOTE: 每个线程加载k_elem_num_per_thread个数据，每个Block加载kBlockN * kBlockK个数据
    for (int i = 0; i < k_load_times_per_thread; ++i)
    {
        // auto k_gmem_head_ptr = gmem_q_ptr + (thread_idx / 2) * seq_stride + ((thread_idx % 2) * 2 + i) * load_elem_size;
        // auto k_smem_head_ptr = &smem_q_ptr[thread_idx][i * load_elem_size];

        // asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
        //             :: "r"(k_gmem_head_ptr),
        //                "l"(k_smem_head_ptr),
        //                "n"(load_elem_size));
    }
    asm volatile("cp.async.commit_group;\n" ::);

    constexpr int n_masking_steps = UP_DIV(kBlockM, kBlockN);

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block)
    {
        constexpr int p_elem_num_per_thread = (kBlockM * kBlockN) / kBlockThreds;
        uint32_t reg_acc_s[p_elem_num_per_thread] = {0};

        // NOTE: 等待异步拷贝完成
        constexpr int N = 0;
        asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
        // NOTE: Block 同步
        __syncthreads();

        // NOTE: Pipeline优化，提前异步加载V数据
        constexpr int v_elem_num_per_thread = (kBlockN * kBlockK) / kBlockThreds;
        const int v_load_times_per_thread = v_elem_num_per_thread / load_elem_size;
    
        // NOTE: 每个线程加载v_elem_num_per_thread个数据，每个Block加载kBlockN * kBlockK个数据
        for (int i = 0; i < v_load_times_per_thread; ++i)
        {
            // auto v_gmem_head_ptr = gmem_v_ptr + (thread_idx / 2) * seq_stride + ((thread_idx % 2) * 2 + i) * load_elem_size;
            // auto v_smem_head_ptr = &smem_v_ptr[thread_idx][i * load_elem_size];

            // asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
            //              :: "r"(v_gmem_head_ptr),
            //                 "l"(v_smem_head_ptr),
            //                 "n"(load_elem_size));
        }

        // NOTE: 提交一个异步拷贝数据指令
        asm volatile("cp.async.commit_group;\n" ::);

        // NOTE: 开始TiledMMA计算Q*K^T
        {
            constexpr int kWarpBlockM = 16;
            constexpr int kWarpBlockN = 8;
            constexpr int kWarpBlockK = 16;

            // 1. 将Thread Block切块使用Tensor Core m16n8k16, 在输出方向切块
            const int kWarps_row = 2;                  // 将warp拆成二维的
            const int kWarps_col = kWarps / kWarps_row;

            // NOTE: MMA with Tensor Core m16n8k16
            constexpr int mma_row_num_per_block = kBlockM / kWarpBlockM;    // 每个warp行方向计算多少数据
            constexpr int mma_col_num_per_block = kBlockN / kWarpBlockN;    // 每个warp列方向计算多少数据
            constexpr int mma_inner_loop_per_block = kBlockK / kWarpBlockK; // 计算一个输出方向warp, Q和K需要循环几次

            const int warp_row = warp_id / 2;
            const int warp_col = (warp_id % 2);
            
            // NOTE: warp对应的offset位置
            const int smem_p_ptr_warp_offset = warp_row * (kBlockM * kBlockN) / 2 + warp_col * (kBlockN / 2);
            const int smem_q_ptr_warp_offset = warp_row * (kBlockM * kBlockK) / 2;
            const int smem_k_ptr_warp_offset = warp_col * (kBlockK / 2);

            // NOTE: 计算一个Block对应的数据block gemm
            for (int i = 0; i < mma_row_num_per_block; ++i)
            {
                for (int j = 0; j < mma_col_num_per_block; ++j)
                {
                    // 1.1 利用切好的块进行Tensor Core运算
                    // NOTE: 引入Double Buffer
                    uint32_t q_reg_warp[2][4];
                    uint32_t k_reg_warp[2][2];
                    auto p_reg_warp = &reg_acc_s;

                    const int smem_p_ptr_block_offset = i * kBlockN + j * kBlockN;

                    for (int t = 0; t < mma_inner_loop_per_block; ++i)
                    {
                        // 1.1.1 加载Q 从Shared Memory To Register File

                        const int smem_q_ptr_warp_offset = warp_row * (kBlockM * kBlockN) / 2 + warp_col * (kBlockN / 2);

                        // uint32_t q_smem_mma_ptr = __cvta_generic_to_shared(&smem_q_ptr[]);
                    }
                }
            }
        }

        // NOTE: 计算一个Block对应的数据softmax数据
        {
            
        }

        // NOTE: 计算Softmax(Q * K^T) * V
        {
            
        }
    }
}

__global__ 
void flash_attention_kernel(const half*  q, 
                            const half*  k, 
                            const half*  v, 
                            half*        o,
                            const int    batch_size,
                            const int    seq_len,
                            const int    num_heads,
                            const int    head_dim)
{
    const int block_index = blockIdx.x; // which block
    const int batch_index = blockIdx.y; // which batch
    const int heads_index = blockIdx.z; // which head

    flash_attention_1block_kernel(q, k, v, o, batch_size, seq_len, num_heads, head_dim, batch_index, heads_index, block_index);
}

void run_flash_fwd(const half*  q, 
                   const half*  k, 
                   const half*  v, 
                   half*        o,
                   const int    batch_size,
                   const int    seq_len,
                   const int    num_heads,
                   const int    head_dim,
                   cudaStream_t stream)
{
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 64;
    constexpr int kNThreads = 32 * 8;
    constexpr int kHeadDim = 128; // NOTE: current support only head_dim == 128

    constexpr size_t smem_size = kBlockM * kHeadDim * sizeof(half) + 2 * kBlockN * kHeadDim * sizeof(half);

    const int num_m_block = seq_len / kBlockM;
    dim3 grid(num_m_block, batch_size, head_dim);

    LOGE("grid: %d, %d, %d, threads: %d",
         num_m_block,
         batch_size,
         head_dim,
         kNThreads);

    auto kernel = &flash_attention_kernel;

    // NOTE: 在sm80中，L1 Cache与Shared Memory共用一块内存，先分配Shared Memory剩下的作为L1 Cache
    if (smem_size >= 48 * 1024)   // 48 kb
    {
        cudaCheck(cudaFuncSetAttribute(kernel,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                smem_size));
    }
    int ctas_per_sm;

    cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                                &ctas_per_sm,
                                kernel,
                                kNThreads,
                                smem_size);

    printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);

    kernel<<<grid, kNThreads, smem_size, stream>>>(q, k, v, o, batch_size, seq_len, num_heads, head_dim);
    afterKernelLaunch();
}
