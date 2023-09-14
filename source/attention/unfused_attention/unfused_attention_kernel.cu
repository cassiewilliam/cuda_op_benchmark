#include "unfused_attention_kernel.hpp"
#include "tensor.hpp"

#include <cmath>

#define FINAL_MASK 0xffffffff

// NOTE: __shfl_xor_sync 具体意义可参考 http://zh0ngtian.tech/posts/ada27037.html
template<typename T>
__inline__ __device__
T warp_reduce_sum(T val)
{
    for (int mask = 16; mask > 0; mask >>=1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);

    return val;
}

// NOTE: 计算一个block之内所有线程访问数据的和
template<typename T>
__inline__ __device__
T block_reduce_sum(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f; // 与上31
    int warp_id = threadIdx.x >> 5; // 除以 32

    val = warp_reduce_sum<T>(val);

    if (lane == 0)
        shared[warp_id] = val;

    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : (T)(0.0f);
    val = warp_reduce_sum<T>(val);

    return val;
}

template<typename T>
__inline__ __device__
T warp_reduce_max(T val)
{
    for (int mask = 16; mask > 0; mask >>=1)
        val = max(val, __shfl_xor_sync(FINAL_MASK, val, mask, 32));
}

// NOTE: 计算一个block中所有数的最大值，将一个block分成32份，每一份32个数据，最多一个block处理1024个数据
template<typename T>
__inline__ __device__
T block_reduce_max(T val)
{
    static __shared__ T shared[32];
    int lane = threadIdx.x & 0x1f; // warp 内线程id
    int warp_id = threadIdx.x >> 5; // warp id

    val = warp_reduce_max(val); // get max elem in each warp

    if (lane == 0) // 记录最大值在对应的shared memory，一个warp只需要lane等于0的线程记录就好
        shared[warp_id] = val;

    __syncthreads();

    // NOTE: 再次计算shared memory中所有元素的最大值，即得到一个block中的最大值
    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e20f;
    val = warp_reduce_max(val);

    return val;
}

// NOTE: 重排输入q,k,v,
// 输入shape为[batch, seq_len, num_heads, head_dim]
// 目标shape为[batch, num_heads, seq_len, head_dim]
template<typename T>
__global__ void rearrange_qkv(void*     src,
                              void*     dst,
                              const int batch_size,
                              const int seq_len,
                              const int num_heads,
                              const int head_dim)
{
    T* src_ptr = static_cast<T*>(src);
    T* dst_ptr = static_cast<T*>(dst);

    int batch_id = blockIdx.x / seq_len;
    int seq_id = blockIdx.x % seq_len;
    int head_id = threadIdx.x / head_dim;
    int id_in_head = threadIdx.x % head_dim;

    int batch_stride = num_heads * seq_len * head_dim;
    int head_stride = seq_len * head_dim;

    int target_id = batch_id * batch_stride + head_id * head_stride + seq_id * head_dim + id_in_head;
    int source_id = threadIdx.x;

    dst_ptr[target_id] = src_ptr[source_id];
}

// NOTE: batched gemm kernel
template <typename TypeA, typename TypeB, typename TypeC, typename TypeAcc>
__global__ void gemm_strided_batched_naive_kernel(bool         trans_a,
                                                  bool         trans_b,
                                                  size_t       m,
                                                  size_t       n,
                                                  size_t       k,
                                                  float        alpha,
                                                  const TypeA* A,
                                                  size_t       lda,
                                                  const TypeB* B,
                                                  size_t       ldb,
                                                  float        beta,
                                                  TypeC*       C,
                                                  size_t       ldc,
                                                  size_t       batch)
{
    // NOTE: 循环batch维度，每次循环batch部数
    for (int bid = blockIdx.x; bid < batch; bid += gridDim.x)
    {
        const TypeA* A_ptr = A + (trans_a ? bid * k * lda : bid * m * lda);
        const TypeB* B_ptr = B + (trans_b ? bid * n * ldb : bid * k * ldb);

        TypeC* C_ptr = C + bid * m * ldc;

           // NOTE: Outter Loop，可进一步并行
        for (size_t m_index = 0; m_index < m; ++m_index)
        {
            // NOTE: Outter Loop，可进一步并行
            for (size_t n_index = threadIdx.x; n_index < n; n += blockDim.x)
            {
                // NOTE: Inner Loop，尽量别并行，存在同步成本
                TypeAcc res = static_cast<TypeAcc>(static_cast<float>(C_ptr[m * ldc + n]) * beta);
                for (size_t k_index = 0; k_index < k; ++k_index)
                {
                    TypeA elem_a = trans_a ? A_ptr[k_index * lda + m_index] : A_ptr[m_index * lda + k_index];
                    TypeB elem_b = trans_b ? B_ptr[n_index * ldb + k_index] : B_ptr[k_index * ldb + n_index];

                    res += static_cast<TypeAcc>(alpha * (float)(elem_a * elem_b));
                }
                C_ptr[m * ldc + n] = res;
            }
        }
    }
}

// NOTE: reference from https://github.com/MegEngine/MegEngine/blob/66b79160d35b2710c00befede0c3fd729109e474/dnn/src/cuda/batched_matrix_mul/naive.cuh
/**
 * @brief C = alpha * A * B + beta * C
 *        A : [b, m, k]
 *        B : [b, n, k], B^T = [b, k, n]
 *        C : [b, m, n]
 */
void gemm_strided_batched_naive(bool         trans_a,
                                bool         trans_b,
                                int          m,
                                int          n,
                                int          k,
                                const void*  alpha,
                                const void*  A,   // 矩阵A的shape: [batch, rows, cols]
                                DataType     A_type,
                                int          lda,        // 矩阵A 行方向的步长
                                size_t       A_stride,   // 矩阵A Batch方向的步长
                                const void*  B,
                                DataType     B_type,
                                int          ldb,
                                size_t       B_stride,
                                const void*  beta,
                                void*        C,
                                DataType     C_type,
                                int          ldc,
                                size_t       C_stride,
                                int          batch,
                                DataType     ACC_type,
                                cudaStream_t stream)
{
    dim3 threads_per_block = dim3(128, 1, 1);
    dim3 block_dimension = dim3(batch, 1, 1);
    float alpha_v = *((float *)alpha);
    float beta_v = *((float *)beta);
    if (A_type == DataType::FLOAT16 && B_type == DataType::FLOAT16 && C_type == DataType::FLOAT16 && ACC_type == DataType::FLOAT32)
    {
        gemm_strided_batched_naive_kernel<half, half, half, float>
            <<<block_dimension, threads_per_block, 0, stream>>>(trans_a,
                                                                trans_b,
                                                                m,
                                                                n,
                                                                k,
                                                                alpha_v,
                                                                (half*)A,
                                                                lda,
                                                                (half*)B,
                                                                ldb,
                                                                beta_v,
                                                                (half*)C,
                                                                ldc,
                                                                batch);
    }
    else
    {
        printf("currently, not support\n");
    }
}

// NOTE: softmax kernel
template<typename T>
__global__
void softmax(void*       qk,
             const void* mask,
             void*       output,
             int         batch_size,
             int         num_heads,
             int         seq_len,
             float       scale)
{
    int batch_id = blockIdx.x / num_heads;
    int qk_offset = blockIdx.x * seq_len * seq_len;
    int mask_offset = batch_id * seq_len * seq_len;

    __shared__ float s_sum, s_max;

    // NOTE: 存在多抛线程可能，边界安全保护
    if (threadIdx.x < seq_len)
    {
        // NOTE: 在行方向计算softmax，即循环seq_len，每一行都是独立
        //       qk的维度为[batch, num_heads, seq_len, seq_len]
        for (int i  = 0; i < seq_len; ++i)
        {
            float qk_value = static_cast<T*>(qk)[threadIdx.x + qk_offset];
            float mask_value = static_cast<const T*>(mask)[threadIdx.x + mask_offset];
            
            // NOTE: mask 为0，1值，将无效位置设置成-10000.0f, 经过softmax后会变成0
            mask_value = (1.0f - mask_value) * -10000.0f;

            float tmp = static_cast<float>(qk_value * scale + mask_value);

            // NOTE: 开始计算safe softmax exp(x_i - x_max) / \sum_i^N(exp(x_i - x_max))

            // 1. 求得当前行的最大值，保存在shared_mem中
            float max_value = block_reduce_max<float>(tmp);

            if (threadIdx.x == 0)
                s_max = max_value;
            __syncthreads();

            // 2. 计算exp(x_i - x_max)
            float exp_value = __expf(tmp - s_max);

            // 3. 计算分母 \sum_i^N(exp(x_i - x_max))
            float sum_value = block_reduce_sum<float>(exp_value);
            if (threadIdx.x == 0)
                s_sum = sum_value + 1e-6f;
            __syncthreads();

            // 4. 计算输出结果
            static_cast<T*>(output)[threadIdx.x + qk_offset] = static_cast<T>(exp_value / s_sum);

            // 5. 处理下一行
            qk_offset += seq_len;
            mask_offset += seq_len;
        }
    }
}

// NOTE: 重排输出矩阵
// 输入shape为[batch, num_heads, seq_len, head_dim]
// 目标shape为[batch, seq_len, num_heads, head_dim]
template<typename T>
__global__ void rearrange_o(void*     src,
                            void*     dst,
                            const int batch_size,
                            const int seq_len,
                            const int num_heads,
                            const int head_dim)
{
    T* src_ptr = static_cast<T*>(src);
    T* dst_ptr = static_cast<T*>(dst);

    const int src_head_stride = seq_len * num_heads;
    int batch_id = blockIdx.x / src_head_stride;
    int seq_id = blockIdx.x % seq_len;
    int head_id = (blockIdx.x % src_head_stride) / seq_len;
    
    const int dst_batch_stride = seq_len * num_heads * head_dim;
    const int dst_seq_stride = num_heads * head_dim;
    int target_id = batch_id * dst_batch_stride + seq_id * dst_seq_stride + head_id * head_dim + threadIdx.x;
    int source_id = blockIdx.x * head_dim + threadIdx.x;

    dst_ptr[target_id] = src_ptr[source_id];
}

// NOTE: o = softmax((q * k^T) / sqrt(head_dim)) * v
//       1. q, k, v重排
//       2. 计算qk = q * k^T
//       3. 计算w = softmax(qk / sqrt(head_dim))
//       4. o_transpose = 计算w * v
//       5. o = 重排(o_transpose)
void unfused_attention_kernel(void*        q,
                              void*        k,
                              void*        v,
                              void*        o,
                              int          batch_size,
                              int          seq_len,
                              int          num_heads,
                              int          head_dim,
                              int          batch_stride,
                              int          row_stride,
                              float        softmax_scale,
                              void*        mask,
                              cudaStream_t stream)
{
    // Step 1: 重排q, k, v 输入shape为[batch, seq_len, num_heads, head_dim]
    //                    目标shape为[batch, num_heads, seq_len, head_dim]
    auto qkv_shape = std::vector<int>{batch_size, num_heads, seq_len, head_dim};
    auto q_buf = Tensor::create(qkv_shape, MemoryType::GPU);
    auto k_buf = Tensor::create(qkv_shape, MemoryType::GPU);
    auto v_buf = Tensor::create(qkv_shape, MemoryType::GPU);
    auto q_ptr = q_buf->data();
    auto k_ptr = k_buf->data();
    auto v_ptr = v_buf->data();
    {
        int m = batch_size * seq_len;
        int n = num_heads * head_dim;

        dim3 grid(m, 1, 1);
        dim3 block(n, 1, 1);
        rearrange_qkv<float><<<grid, block, 0, stream>>>(q, q_ptr, batch_size, seq_len, num_heads, head_dim);
        rearrange_qkv<float><<<grid, block, 0, stream>>>(k, k_ptr, batch_size, seq_len, num_heads, head_dim);
        rearrange_qkv<float><<<grid, block, 0, stream>>>(v, v_ptr, batch_size, seq_len, num_heads, head_dim);
    }

    // Step 2: 计算 qk = (q * k^T) / sqrt(head_dim): r = alpha * q * k^T
    auto qk = Tensor::create(std::vector<int>{batch_size, num_heads, seq_len, seq_len}, MemoryType::GPU);
    auto qk_ptr = qk->data();
    {
        float alpha = 1.0f, beta = 0.0f;
        gemm_strided_batched_naive(false, true, 
                                   seq_len, seq_len, head_dim, 
                                   &alpha, 
                                   q_ptr, DataType::FLOAT32, head_dim, seq_len * head_dim,
                                   k_ptr, DataType::FLOAT32, head_dim, seq_len * head_dim,
                                   &beta,
                                   qk_ptr, DataType::FLOAT32, seq_len, seq_len * seq_len,
                                   batch_size * num_heads,
                                   DataType::FLOAT32,
                                   stream);

    }

    // Step 3: 计算 w = softmax(r, dim=-1)
    auto w = Tensor::create(std::vector<int>{batch_size, num_heads, seq_len, seq_len}, MemoryType::GPU);
    auto w_ptr = w->data();
    {
        dim3 grid, block;
        // NOTE: 按照2的指数进行抛线程
        block.x = std::min(1024, std::max(32, (int)std::pow(2, (std::ceil(std::log2(seq_len))))));
        grid.x = batch_size * num_heads;

        // NOTE: 一次计算一行seq_len个softmax
        softmax<float><<<grid, block, 0, stream>>>(qk_ptr, mask, w_ptr, batch_size, num_heads, seq_len, softmax_scale);
    }

    // Step 4: 计算 o = w * v
    auto o_buf = Tensor::create(qkv_shape, MemoryType::GPU);
    auto o_ptr = o_buf->data();
    {
        float alpha = 1.0f, beta = 0.0f;
        gemm_strided_batched_naive(false, true, 
                                   seq_len, seq_len, head_dim, 
                                   &alpha, 
                                   w_ptr, DataType::FLOAT32, seq_len, seq_len * seq_len,
                                   v_ptr, DataType::FLOAT32, head_dim, seq_len * head_dim,
                                   &beta,
                                   o_ptr, DataType::FLOAT32, head_dim, seq_len * head_dim,
                                   batch_size * num_heads,
                                   DataType::FLOAT32,
                                   stream);
    }

    // Step 5: 重排o 输入shape为[batch, num_heads, seq_len, head_dim]
    //               目标shape为[batch, seq_len, num_heads, head_dim]
    {
        dim3 grid, block;
        grid.x = batch_size * num_heads * seq_len;
        block.x = head_dim;

        rearrange_o<float><<<grid, block, 0, stream>>>(o_ptr, o, batch_size, seq_len, num_heads, head_dim);
    }
}