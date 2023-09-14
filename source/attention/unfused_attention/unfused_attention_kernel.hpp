#pragma once

#include "cuda_runtime.hpp"
#include "enum.hpp"

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
                              cudaStream_t stream);