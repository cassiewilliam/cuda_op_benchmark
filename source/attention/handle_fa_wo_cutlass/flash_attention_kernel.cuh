#pragma once

#include "defines.hpp"
#include "cuda_runtime.hpp"

void run_flash_fwd(const half*  q, 
                   const half*  k, 
                   const half*  v, 
                   half*        o,
                   const int    batch_size,
                   const int    seq_len,
                   const int    num_heads,
                   const int    head_dim,
                   cudaStream_t stream);
