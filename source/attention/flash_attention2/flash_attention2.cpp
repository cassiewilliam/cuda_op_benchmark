#include "flash.h"
#include "flash_attention2.hpp"
#include "cuda_runtime.hpp"

#include <cutlass/numeric_types.h>

void flash_attention2(const Tensor &qkv,         // total_n x num_heads x head_size * 3, total_n := \sum_{i=0}^{b} s_i
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

    Flash_fwd_params params;

    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = static_cast<__half *>(q.data());
    params.k_ptr = static_cast<__half *>(k.data());
    params.v_ptr = static_cast<__half *>(v.data());

    params.q_row_stride = num_heads * size_per_head;
    params.q_head_stride = size_per_head;

    params.k_row_stride = num_heads * size_per_head;
    params.k_head_stride = size_per_head;

    params.v_row_stride = num_heads * size_per_head;
    params.v_head_stride = size_per_head;

    params.o_ptr = static_cast<__half *>(out.data());

    params.o_row_stride = num_heads * size_per_head;
    params.o_head_stride = size_per_head;

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens.data());
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens.data());

    params.window_size_left = -1;
    params.window_size_right = -1;

    // Set the dimensions.
    params.b = batch_size;
    params.h = num_heads;
    params.d = size_per_head;
    params.seqlen_q = max_seqlen;
    params.seqlen_k = max_seqlen;

    params.p_dropout = 1.0f;
    params.seqlen_q_rounded = max_seqlen;
    params.seqlen_k_rounded = max_seqlen;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    params.h_h_k_ratio = 1;

    auto stream = CUDARuntime::GetInstance()->getCurrentCUDAStream();

    run_mha_fwd_<cutlass::half_t, 64>(params, stream);
}