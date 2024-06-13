#include "ft_bert_attention.hpp"
#include "open_attention.h"

void ft_bert_attention(const Tensor&  q,      // batch_size x seqlen_q x num_heads   x head_size
                       const Tensor&  k,      // batch_size x seqlen_k x num_heads_k x head_size
                       const Tensor&  v,      // batch_size x seqlen_k x num_heads_k x head_size
                       const Tensor&  q_bias, // batch_size x seqlen_k x num_heads_k x head_size
                       const Tensor&  k_bias, // batch_size x seqlen_k x num_heads_k x head_size
                       const Tensor&  v_bias, // batch_size x seqlen_k x num_heads_k x head_size
                       const Tensor&  mask,
                       Tensor&        out,    // batch_size x seqlen_q x num_heads   x head_size
                       Tensor&        buffer,
                       cudaStream_t   stream,
                       cublasHandle_t handle)
{
    CE_CHECK(k.dtype() == q.dtype(), "query and key must have the same dtype");
    CE_CHECK(v.dtype() == q.dtype(), "query and value must have the same dtype");

    CE_CHECK(q.isCUDA(), "Input tensor must be on CUDA device");
    CE_CHECK(k.isCUDA(), "Input tensor must be on CUDA device");
    CE_CHECK(v.isCUDA(), "Input tensor must be on CUDA device");

    CE_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CE_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CE_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    // NOTE: 输入Tensor shape: [batch, seq len, num heads, head dims]
    const auto sizes = q.shapes();

    const int batch_size = sizes[0];
    const int seqlen     = sizes[1];
    const int num_heads  = sizes[2];
    const int head_size  = sizes[3];   // head dimension

    LOGE("bshd: %d, %d, %d, %d",
         batch_size,
         seqlen,
         num_heads,
         head_size);

    auto q_ptr = (__half *)q.data();
    auto k_ptr = (__half *)k.data();
    auto v_ptr = (__half *)v.data();
    auto q_bias_ptr = (__half *)q_bias.data();
    auto k_bias_ptr = (__half *)k_bias.data();
    auto v_bias_ptr = (__half *)v_bias.data();
    auto o_ptr = (__half *)out.data();
    auto mask_ptr = (__half *)mask.data();
    auto buf_ptr = (__half *)buffer.data();

    const float softmax_scale = 1.0f / std::sqrt(head_size);
    fastertransformer::mha_nofuse_kernel(stream,
                                         handle,
                                         q_ptr,
                                         q_bias_ptr,
                                         k_ptr,
                                         k_bias_ptr,
                                         v_ptr,
                                         v_bias_ptr,
                                         mask_ptr,
                                         o_ptr,
                                         buf_ptr,
                                         batch_size,
                                         seqlen,
                                         num_heads,
                                         head_size,
                                         (half)(softmax_scale));
     afterKernelLaunch();
}