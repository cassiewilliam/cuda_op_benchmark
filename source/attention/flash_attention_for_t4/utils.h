/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once


#include <cuda_runtime.h>

#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cmath>
#include <cuda_fp16.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace flash {

using namespace cute;

struct BlockInfo
{

    template<typename Params>
    __device__ BlockInfo(const Params& params, const int bidb)
        : sum_s(params.cu_seqlens[bidb])
        , actual_seqlen(params.cu_seqlens[bidb + 1] - sum_s)
    {}

    template<typename index_t>
    inline __device__ index_t qkv_offset(const index_t row_stride, const int bidb) const
    {
        return uint32_t(sum_s) * row_stride;
    }

    template<typename index_t>
    inline __device__ index_t o_offset(const index_t row_stride, const int bidb) const
    {
        return uint32_t(sum_s) * row_stride;
    }
    const int sum_s;
    const int actual_seqlen;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ uint32_t relu2(const uint32_t x);

template<>
inline __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x)
{
    uint32_t       res;
    const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
    asm volatile("{\n"
                 "\t .reg .f16x2 sela;\n"
                 "\t set.gtu.u32.f16x2 sela, %1, %2;\n"
                 "\t and.b32 %0, sela, %1;\n"
                 "}\n"
                 : "=r"(res)
                 : "r"(x), "r"(zero));
#endif
    return res;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<>
inline __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x)
{
    uint32_t       res;
    const uint32_t zero = 0u;
    asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template<typename T>
inline __device__ uint32_t convert_relu2(const float2 x);

template<>
inline __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x)
{
    uint32_t       res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

template<>
inline __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x)
{
    uint32_t       res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp
{
    __device__ inline T operator()(T const& x, T const& y)
    {
        return x > y ? x : y;
    }
};

template<>
struct MaxOp<float>
{
    // This is slightly faster
    __device__ inline float operator()(float const& x, float const& y)
    {
        return max(x, y);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp
{
    __device__ inline T operator()(T const& x, T const& y)
    {
        return x + y;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce
{
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator& op)
    {
        constexpr int OFFSET = THREADS / 2;
        x                    = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2>
{
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator& op)
    {
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
        return x;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs = false,
         bool B_in_regs = false,
         typename Tensor0,
         typename Tensor1,
         typename Tensor2,
         typename Tensor3,
         typename Tensor4,
         typename TiledMma,
         typename TiledCopyA,
         typename TiledCopyB,
         typename ThrCopyA,
         typename ThrCopyB>
inline __device__ void gemm(Tensor0&       acc,
                            Tensor1&       tCrA,
                            Tensor2&       tCrB,
                            Tensor3 const& tCsA,
                            Tensor4 const& tCsB,
                            TiledMma       tiled_mma,
                            TiledCopyA     smem_tiled_copy_A,
                            TiledCopyB     smem_tiled_copy_B,
                            ThrCopyA       smem_thr_copy_A,
                            ThrCopyB       smem_thr_copy_B)
{
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));    // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));    // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));   // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));   // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));   // N
    if (!A_in_regs)
    {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    }
    if (!B_in_regs)
    {
        cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    }
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i)
    {
        if (i < size<2>(tCrA) - 1)
        {
            if (!A_in_regs)
            {
                cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            }
            if (!B_in_regs)
            {
                cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
            }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0,
         typename Tensor1,
         typename Tensor2,
         typename Tensor3,
         typename TiledMma,
         typename TiledCopy,
         typename ThrCopy>
inline __device__ void gemm_A_in_regs(Tensor0&       acc,
                                      Tensor1&       tCrA,
                                      Tensor2&       tCrB,
                                      Tensor3 const& tCsB,
                                      TiledMma       tiled_mma,
                                      TiledCopy      smem_tiled_copy_B,
                                      ThrCopy        smem_thr_copy_B)
{
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));    // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));    // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));   // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));   // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i)
    {
        if (i < size<2>(tCrA) - 1)
        {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout)
{
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});   // ((2, 2), MMA_M, MMA_N)
    // TD [2023-08-13]: Idk why but get<0, 1>(l) doesn't work for Cutlass 3.2, I'm getting
    // "int_tuple.hpp(74): error: conversion to inaccessible base class"
    // return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l),
    // get<2>(l)));
    return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)),
                       make_layout(get<0>(get<0>(l)), get<2>(l)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert rowcol_layout from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
template<typename MMA_traits, typename Layout>
inline __device__ auto convert_layout_rowcol_Aregs(Layout rowcol_layout)
{
    using X = Underscore;
    static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
    static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    constexpr int MMA_N_divisor = mma_shape_K == 8 ? 1 : 2;
    auto          l             = logical_divide(
        rowcol_layout,
        Shape<X, Shape<X, Int<MMA_N_divisor>>>{});   // ((2, MMA_M), (2, (2, MMA_N / 2)))
    // TD [2023-08-13]: Same error as above on Cutlass 3.2
    // return make_layout(make_layout(get<1, 0>(l), get<0, 0>(l), get<1, 1, 0>(l)),
    //                    get<0, 1>(l),
    //                    get<1, 1, 1>(l));
    return make_layout(make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)), get<0>(get<1>(get<1>(l)))),
                       get<1>(get<0>(l)),
                       get<1>(get<1>(get<1>(l))));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename To_type, typename Engine, typename Layout>
inline __device__ auto convert_type(Tensor<Engine, Layout> const& tensor)
{
    using From_type                                                 = typename Engine::value_type;
    constexpr int                                             numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(
        *reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}


////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template<int N>
CUTE_HOST_DEVICE void cp_async_wait()
{
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Clear_OOB_MN = false,
         typename TiledCopy,
         typename Engine0,
         typename Layout0,
         typename Engine1,
         typename Layout1,
         typename Engine2,
         typename Layout2>
inline __device__ void copy(TiledCopy                       tiled_copy,
                            Tensor<Engine0, Layout0> const& S,
                            Tensor<Engine1, Layout1>&       D,
                            Tensor<Engine2, Layout2> const& identity_MN,
                            const int                       max_MN = 0)
{
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));   // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));   // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));   // MMA_K

#pragma unroll
    for (int m = 0; m < size<1>(S); ++m)
    {
        if (!Clear_OOB_MN || get<0>(identity_MN(0, m, 0)) < max_MN)
        {
#pragma unroll
            for (int k = 0; k < size<2>(S); ++k)
            {
                cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
            }
        }
        else if (Clear_OOB_MN)
        {
            cute::clear(D(_, m, _));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool zero_init = true,
         typename Engine0,
         typename Layout0,
         typename Engine1,
         typename Layout1,
         typename Operator>
__device__ inline void thread_reduce_(Tensor<Engine0, Layout0> const& tensor,
                                      Tensor<Engine1, Layout1>&       summary,
                                      Operator&                       op)
{
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
#pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++)
    {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
#pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++)
        {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void quad_allreduce_(Tensor<Engine0, Layout0>& dst,
                                       Tensor<Engine1, Layout1>& src,
                                       Operator&                 op)
{
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
#pragma unroll
    for (int i = 0; i < size(dst); i++)
    {
        dst(i) = Allreduce<4>::run(src(i), op);
    }
}

template<bool zero_init = true,
         typename Engine0,
         typename Layout0,
         typename Engine1,
         typename Layout1,
         typename Operator>
__device__ inline void reduce_(Tensor<Engine0, Layout0> const& tensor,
                               Tensor<Engine1, Layout1>&       summary,
                               Operator&                       op)
{
    thread_reduce_<zero_init>(tensor, summary, op);
    quad_allreduce_(summary, summary, op);
}

template<bool zero_init = true,
         typename Engine0,
         typename Layout0,
         typename Engine1,
         typename Layout1>
__device__ inline void reduce_max(Tensor<Engine0, Layout0> const& tensor,
                                  Tensor<Engine1, Layout1>&       max)
{
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_sum(Tensor<Engine0, Layout0> const& tensor,
                                  Tensor<Engine1, Layout1>&       sum)
{
    SumOp<float> sum_op;
    reduce_(tensor, sum, sum_op);
}

// Apply the exp to all the elements.
template<bool Scale_max = true,
         typename Engine0,
         typename Layout0,
         typename Engine1,
         typename Layout1>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0>&       tensor,
                                        Tensor<Engine1, Layout1> const& max,
                                        const float                     scale)
{
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
#pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi)
    {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY
                                     ? 0.f
                                     : max(mi) * (Scale_max ? scale : float(M_LOG2E));
#pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)
        {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

template<typename Engine, typename Layout>
inline __device__ void apply_mask(Tensor<Engine, Layout>& tensor,
                                  const int               max_seqlen_k,
                                  const int               col_idx_offset_ = 0)
{
    // tensor has shape (ncol=(2, MMA_M), nrow=(2, MMA_N))
    static_assert(Layout::rank == 2, "Only support 2D Tensor");
    const int lane_id        = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int nj = 0; nj < size<1, 1>(tensor); ++nj)
    {
        const int col_idx_base = col_idx_offset + nj * 8;
        #pragma unroll
        for (int j = 0; j < size<1, 0>(tensor); ++j)
        {
            const int col_idx = col_idx_base + j;
            if (col_idx >= max_seqlen_k)
            {
                // Without the "make_coord" we get wrong results
                #pragma unroll
                for (int mi = 0; mi < size<0>(tensor); ++mi)
                {
                    tensor(mi, make_coord(j, nj)) = -INFINITY;
                }
            }
        }
    }
}

}   // namespace flash
