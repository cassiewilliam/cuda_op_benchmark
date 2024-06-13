#include "flash_attn_kernel.cuh"
#include "flash_byte_attention.hpp"
#include "cuda_runtime.hpp"

#include <cutlass/numeric_types.h>

void flash_byte_attention(const Tensor &qkv,         // total_n x num_heads x head_size * 3, total_n := \sum_{i=0}^{b} s_i
                          const Tensor &q,           // total_n x num_heads x head_size, total_n := \sum_{i=0}^{b} s_i
                          const Tensor &k,           // total_n x num_heads x head_size, total_n := \sum_{i=0}^{b} s_i
                          const Tensor &v,           // total_n x num_heads x head_size, total_n := \sum_{i=0}^{b} s_i
                          const Tensor &mask,        // batch_size, seq_len x seq_len
                          Tensor       &out,         // total_n x num_heads x head_size, total_n := \sum_{i=0}^{b} s_i
                          const Tensor &cu_seqlens,  // b+1 elements
                          const int     max_seqlen)
{
    CE_CHECK(cu_seqlens.dtype() == DataType::INT32, "cu_seqlens must have dtype int32");

    const auto sizes = qkv.shapes();

    const int batch_size = cu_seqlens.size() - 1;
    const int num_heads = sizes[2];
    const int size_per_head = sizes[3];

    CE_CHECK(batch_size > 0, "batch size must be positive");

    const float softmax_scale = 1.0f / std::sqrt(size_per_head);

    flash::AttnParams params;

    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.qkv_ptr = static_cast<__half *>(qkv.data());

    bool use_split_qkv = false;
    if (use_split_qkv)
    {
        params.q_ptr = static_cast<__half *>(q.data());
        params.k_ptr = static_cast<__half *>(k.data());
        params.v_ptr = static_cast<__half *>(v.data());
        params.qkv_row_stride =  num_heads * size_per_head;
    }
    else
    {
        params.q_ptr = static_cast<__half *>(qkv.data());
        params.k_ptr = static_cast<__half *>(qkv.data()) + num_heads * size_per_head;
        params.v_ptr = static_cast<__half *>(qkv.data()) + num_heads * size_per_head * 2;
        params.qkv_row_stride =  num_heads * size_per_head * 3;
    }

    // All stride are in elements, not bytes.
    params.bias_batch_stride = num_heads * size_per_head;
    params.row_stride = num_heads * size_per_head;
    params.head_stride = size_per_head;

    params.o_ptr = static_cast<__half *>(out.data());

    params.cu_seqlens = static_cast<int *>(cu_seqlens.data());

    // Set the dimensions.
    params.b = batch_size;
    params.s = max_seqlen;
    params.h = num_heads;
    params.d = size_per_head;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    auto stream = CUDARuntime::GetInstance()->getCurrentCUDAStream();

    run_mha_fwd_<cutlass::half_t>(params, stream);
}