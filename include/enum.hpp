#pragma once

#include "defines.hpp"

enum OpType
{
    OpType_Reduce = 0,
    OpType_Gemm,
    OpType_Attention,
    OpType_LayerNorm,
    OpType_Softmax,
    OpType_Conv3x3,
};

enum KernelType
{
    KT_Default = 0,
    KT_ByteTransformerAttention,
    KT_FlashAttention_For_T4,
    KT_FlashAttention2,
    KT_HandleFlashAttentionWithoutCutlass,
    KT_UnFusedAttention,
    KT_UnFusedAttentionCublas

    KT_HGEMM_Naive,
    KT_HGEMM_WMMA,
    KT_HGEMM_MMA_PTX,
    KT_HGEMM_MMA_PTX_OPT,
    KT_HGEMM_CUTLASS,
};

enum ErrorCode
{
#ifdef NO_ERROR
#undef NO_ERROR
#endif // NO_ERROR
    NO_ERROR           = 0,
    OUT_OF_MEMORY      = 1,
    NOT_SUPPORT        = 2,
    COMPUTE_SIZE_ERROR = 3,
    NO_EXECUTION       = 4,
    INVALID_VALUE      = 5,
};

enum DataType
{
    UNKNOWN = -1,
    INT64   = 0,
    UINT64,
    INT32,
    UINT32,
    INT16,
    UINT16,
    INT8,
    UINT8,
    FLOAT64,
    FLOAT32,
    FLOAT16,
    BFLOAT16,
};

enum DimFormat
{
    NHWC   = 0,
    NCHW   = 1,
    NC4HW4 = 2,
};

enum MemoryType
{
    CPU  = 0,   // CPU Memory
    GPU  = 1,   // GPU Global Memory
    SMEM = 2,   // GPU Shared Memory
};
