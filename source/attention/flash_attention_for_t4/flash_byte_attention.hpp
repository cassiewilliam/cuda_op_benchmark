
#pragma once

#include "tensor.hpp"

void flash_byte_attention(const Tensor &qkv,         // total_n x num_heads x head_size * 3, total_n := \sum_{i=0}^{b} s_i
                          const Tensor &q,           // total_n x num_heads x head_size, total_n := \sum_{i=0}^{b} s_i
                          const Tensor &k,           // total_n x num_heads x head_size, total_n := \sum_{i=0}^{b} s_i
                          const Tensor &v,           // total_n x num_heads x head_size, total_n := \sum_{i=0}^{b} s_i
                          const Tensor &mask,        // batch_size, seq_len x seq_len
                          Tensor       &out,         // total_n x num_heads x head_size, total_n := \sum_{i=0}^{b} s_i
                          const Tensor &cu_seqlens,  // b+1 elements
                          const int     max_seqlen);