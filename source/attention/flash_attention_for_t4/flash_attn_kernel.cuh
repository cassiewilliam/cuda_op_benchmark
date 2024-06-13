/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once


#include <cute/algorithm/copy.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/layout.h>

#include <cuda.h>
#include <vector>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

namespace flash
{
using namespace cute;
////////////////////////////////////////////////////////////////////////////////////////////////////

struct AttnParams 
{
    // The dimensions.
    int b;
    int s;
    int h;
    int d;

    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The QKV matrices.
    void *__restrict__ qkv_ptr;

    // The O matrix (output).
    void * __restrict__ o_ptr;
    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens;

    // The stride between rows of the Q, K and V matrices.
    uint32_t qkv_row_stride;

    uint32_t row_stride;
    uint32_t head_stride;

    uint32_t bias_batch_stride;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;
};


template<int  kHeadDim_,
         int  kBlockM_,
         int  kBlockN_,
         int  kNWarps_,
         typename elem_type = cutlass::half_t>
struct Kernel_Traits
{
    using Element      = cutlass::half_t;
    using ElementAccum = float;
    using index_t      = uint32_t;

    using MMA_Atom_Arch = MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>;
    using ValLayoutMNK  = Layout<Shape<_1, _2, _2>>;

    using SmemCopyAtom           = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;

    // The number of threads.
    static constexpr int kNWarps   = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM  = kBlockM_;
    static constexpr int kBlockN  = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);

    static constexpr int kBlockKSmem = 64;
    static constexpr int kBlockKGmem = 64;
    static constexpr int kSwizzle    = 3;

    using TiledMma = TiledMMA<MMA_Atom_Arch,
                              Layout<Shape<Int<kNWarps>, _1, _1>>,   // 2x1x1
                              ValLayoutMNK>;   // 1x2x1 or 1x2x2 value group for
                                               // 16x16x16 MMA and LDSM
    using SmemLayoutAtomQ = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        // This has to be kBlockKSmem, using kHeadDim gives wrong results for d=128
        Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{},
                                               Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtomQ{},
                                                Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    // This has to be kBlockN and not 8, otherwise we get wrong results for d=128
    using SmemLayoutAtomVtransposedNoSwizzle = Layout<Shape<Int<kBlockKSmem>, Int<kBlockN>>,
                                                      Stride<_1, Int<kBlockKSmem>>>;

    using SmemLayoutAtomVtransposed = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                                                           SmemLayoutAtomVtransposedNoSwizzle{}));
    using SmemLayoutVtransposed     = decltype(tile_to_shape(SmemLayoutAtomVtransposed{},
                                                         Shape<Int<kHeadDim>, Int<kBlockN>>{}));

    // Maybe the VtransposeNoSwizzle just needs to have the right shape
    // And the strides don't matter?
    using SmemLayoutVtransposedNoSwizzle = decltype(tile_to_shape(
        SmemLayoutAtomVtransposedNoSwizzle{},
        Shape<Int<kHeadDim>, Int<kBlockN>>{}));
    // using SmemLayoutVtransposedNoSwizzle = decltype(SmemLayoutVtransposed{}.layout_fn());

    using SmemLayoutAtomO = decltype(composition(
        Swizzle<kSwizzle, 3, 3>{},
        Layout<Shape<Int<8>, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutO     = decltype(tile_to_shape(SmemLayoutAtomO{},
                                               Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemCopyAtomO      = Copy_Atom<DefaultCopy, Element>;
    using SmemCopyAtomOaccum = Copy_Atom<DefaultCopy, ElementAccum>;

    static constexpr int kSmemQCount  = size(SmemLayoutQ{});
    static constexpr int kSmemKVCount = size(SmemLayoutKV{}) * 2;
    static constexpr int kSmemQSize   = kSmemQCount * sizeof(Element);
    static constexpr int kSmemKVSize  = kSmemKVCount * sizeof(Element);
    static constexpr int kSmemSize    = kSmemQSize + kSmemKVSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0,
                  "kHeadDim must be a multiple of kGmemElemsPerLoad");
    // Using kBlockKSmem here is 6-10% faster than kBlockKGmem for d=128 because of bank conflicts.
    // For example, for d=128, smem is split into 2 "pages", each page takes care of columns
    // 0-63 and 64-127. If we have 16 threads per row for gmem read, when we write to smem,
    // thread 0 - 7 will write to the first page and thread 8 - 15 will write to the second page,
    // to the same banks.
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0,
                  "kNThreads must be a multiple of kGmemThreadsPerRow");

    using GmemLayoutAtom = Layout<
        Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;

    using GmemTiledCopyQKV = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        GmemLayoutAtom{},
        Layout<Shape<_1, _8>>{}));   // Val layout, 8 vals per read

    using GmemTiledCopyO = decltype(make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        GmemLayoutAtom{},
        Layout<Shape<_1, _8>>{}));   // Val layout, 8 vals per store
};

};

template<typename T>
void run_mha_fwd_hdim64(flash::AttnParams& params, cudaStream_t stream);

// NOTE: 最终对外接口
////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
void run_mha_fwd_(flash::AttnParams &params, cudaStream_t stream);