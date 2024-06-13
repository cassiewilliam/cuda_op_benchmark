#include "mem_effi_attention.hpp"


std::vector<Tensor> mem_effi_attention(const Tensor& q,
                                      const Tensor& k,
                                      const Tensor& v,
                                      Tensor&       out,
                                      const float   p_dropout,
                                      const float   softmax_scale,
                                      const bool    is_causal,
                                      const bool    return_softmax,
                                      const Tensor& mask,
                                      cudaStream_t  stream)
{
    LOGE("raw attention begin");

    CE_CHECK(k.dtype() == q.dtype(), "query and key must have the same dtype");
    CE_CHECK(v.dtype() == q.dtype(), "query and value must have the same dtype");

    CE_CHECK(q.isCUDA(), "Input tensor must be on CUDA device");
    CE_CHECK(k.isCUDA(), "Input tensor must be on CUDA device");
    CE_CHECK(v.isCUDA(), "Input tensor must be on CUDA device");

    CE_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CE_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    CE_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    LOGE("q, stride: %ld, %d, %d, %d, %d",
         q.strides().size(),
         q.stride(0),
         q.stride(-3),
         q.stride(-2),
         q.stride(-1));

    // NOTE: 输入Tensor shape: [batch, seq len, num heads, head dims]
    const auto sizes = q.shapes();

    const int batch_size = sizes[0];
    const int seqlen     = sizes[1];
    const int num_heads  = sizes[2];
    const int head_size  = sizes[3];   // head dimension

    CE_CHECK(head_size % 8 == 0, "Not Support Head Dimension");

    // auto stream = runtime->getCurrentCUDAStream();

    // unfused_attention_kernel(q.data(),
    //                          k.data(),
    //                          v.data(),
    //                          out.data(),
    //                          batch_size,
    //                          seqlen,
    //                          num_heads,
    //                          head_size,
    //                          q.stride(0),
    //                          q.stride(1),
    //                          softmax_scale,
    //                          mask.data(),
    //                          stream);

    return {};
}