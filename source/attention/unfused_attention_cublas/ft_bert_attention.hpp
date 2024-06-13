#pragma once

#include "cuda_runtime.hpp"
#include "tensor.hpp"

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
                       cublasHandle_t handle);