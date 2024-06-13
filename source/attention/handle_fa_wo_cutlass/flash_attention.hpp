#pragma once

#include "cuda_runtime.hpp"
#include "tensor.hpp"


std::vector<Tensor> flash_attention(
                const Tensor& q,            // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                const Tensor& k,            // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                const Tensor& v,            // total_k x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                Tensor&       out,          // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
                const int     max_seqlen,   // batch中最大的的序列长度
                const float   softmax_scale,
                const bool    zero_tensors,
                const bool    is_causal,
                const int     num_splits);