/*
 * Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * Open sourced multi-head attention
 **/

#pragma once

#include "common.h"

#include <assert.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>


namespace fastertransformer{
  
void 
mha_nofuse_kernel(cudaStream_t stream,
                  cublasHandle_t handle,
                  __half* Q,
                  const __half* bias_Q,
                  __half* K,
                  const __half* bias_K,
                  __half* V,
                  const __half* bias_V,
                  const __half* attr_mask,
                  __half* dst,
                  __half* buffer,
                  const int batch_size,
                  const int seq_len,
                  const int head_num,
                  const int size_per_head,
                  const __half scaler);

}//namespace fastertransformer
