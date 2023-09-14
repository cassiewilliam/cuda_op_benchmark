#pragma once

#include "cuda_runtime.hpp"
#include "tensor.hpp"

std::vector<Tensor> unfused_attention(
    const Tensor& q,     // batch_size x seqlen_q x num_heads   x head_size
    const Tensor& k,     // batch_size x seqlen_k x num_heads_k x head_size
    const Tensor& v,     // batch_size x seqlen_k x num_heads_k x head_size
    Tensor&       out,   // batch_size x seqlen_q x num_heads   x head_size
    const float   p_dropout,
    const float   softmax_scale,
    const bool    is_causal,
    const bool    return_softmax,
    const Tensor& mask,
    cudaStream_t  stream);