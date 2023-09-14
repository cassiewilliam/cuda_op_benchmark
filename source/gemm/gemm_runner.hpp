#pragma once

#include "gemm_runner.hpp"
#include "enum.hpp"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/*
 * 问题定义: A(m, k) * B(k, n) = C(m, n)
 */
void gemm_runner(bool         transa,
                 bool         transb,
                 int          m,
                 int          n,
                 int          k,
                 const float *alpha,
                 const void * A,
                 DataType     Atype,
                 const void * B,
                 DataType     Btype,
                 const float *beta,
                 const void * C,
                 DataType     Ctype,
                 void *       D,
                 DataType     Dtype,
                 KernelType   algoType);
