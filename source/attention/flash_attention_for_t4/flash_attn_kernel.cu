#include "flash_attn_kernel.cuh"
#include "cuda_runtime.hpp"
#include "utils.h"

namespace flash 
{

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Is_first,
         bool Check_inf = false,
         typename Tensor0,
         typename Tensor1,
         typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0& scores,
                                         Tensor1& scores_max,
                                         Tensor1& scores_sum,
                                         Tensor2& acc_o,
                                         float    softmax_scale_log2)
{
    if (Is_first)
    {
        flash::template reduce_max</*zero_init=*/true>(scores, scores_max);
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        flash::reduce_sum(scores, scores_sum);
    }
    else
    {
        Tensor scores_max_prev = make_fragment_like(scores_max);
        cute::copy(scores_max, scores_max_prev);
        flash::template reduce_max</*zero_init=*/false>(scores, scores_max);
        // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));

        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi)
        {
            float scores_max_cur = !Check_inf
                                        ? scores_max(mi)
                                        : (scores_max(mi) == -INFINITY ? 0.0f : scores_max(mi));
            float scores_scale = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
            scores_sum(mi) *= scores_scale;
            
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni)
            {
                acc_o_rowcol(mi, ni) *= scores_scale;
            }
        }
        
        flash::scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        Tensor scores_sum_cur = make_fragment_like(scores_sum);
        flash::reduce_sum(scores, scores_sum_cur);
        
        #pragma unroll
        for (int mi = 0; mi < size(scores_sum); ++mi)
        {
            scores_sum(mi) += scores_sum_cur(mi);
        }
    }
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename Params>
inline __device__ void compute_attn_1rowblock(const Params& params,
                                              const int     bidb,
                                              const int     bidh,
                                              const int     m_block)
{
    using Element      = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t      = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM  = Kernel_traits::kBlockM;  // 32
    constexpr int kBlockN  = Kernel_traits::kBlockN;  // 32
    constexpr int kHeadDim = Kernel_traits::kHeadDim; // 64
    constexpr int kNWarps  = Kernel_traits::kNWarps;  // 2
    constexpr int MMA_M    = kBlockM / decltype(size<0>(typename Kernel_traits::TiledMma::TiledShape_MNK{}))::value; // 1

    // TiledMma: MNK => 32x16x16

    // NOTE: Batch中不同的序列具有不同的长度，最小计算单元为kBlockM
    const BlockInfo binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen || binfo.actual_seqlen == 0)
        return;

    int n_block_max = cute::ceil_div(binfo.actual_seqlen, kBlockN);

    const index_t row_offset_q = binfo.qkv_offset(params.qkv_row_stride, bidb) +
                                    m_block * kBlockM * params.qkv_row_stride +
                                    bidh * params.head_stride;

    // NOTE: 从后往前算，最后一个block可能存在需要按照Block填充的情况
    const index_t row_offset_k = binfo.qkv_offset(params.qkv_row_stride, bidb) +
                                    (n_block_max - 1) * kBlockN * params.qkv_row_stride +
                                    bidh * params.head_stride;

    const index_t row_offset_v = binfo.qkv_offset(params.qkv_row_stride, bidb) +
                                    (n_block_max - 1) * kBlockN * params.qkv_row_stride +
                                    bidh * params.head_stride;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.qkv_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.qkv_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.qkv_row_stride, _1{}));

    // printf("%d, %d, %d, %d, %d, %d\n", bidb, m_block, bidh, row_offset_q, row_offset_k, row_offset_v);

    Tensor sQ           = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),typename Kernel_traits::SmemLayoutQ{});
    Tensor sK           = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV           = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt          = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);   // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);   // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // if (cute::thread0())
    // {
    //     printf("actual_seqlen: %d, %d\n", binfo.actual_seqlen, (n_block_max - 1) * kBlockN);
    //     print(tQgQ);
    //     printf("\n");
    //     for (int i = 0; i < size(tQgQ); ++i) {
    //         printf("%d, %.4f\n", i, (float)tQgQ[i]);
    //     }

    //     print(tKgK);
    //     printf("\n");
    //     for (int i = 0; i < size(tKgK); ++i) {
    //         printf("%d, %.4f\n", i, (float)tKgK[i]);
    //     }
    // }

    typename Kernel_traits::TiledMma tiled_mma;
    auto   thr_mma = tiled_mma.get_thread_slice(tidx);

    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);             // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);             // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);   // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});   // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //
    auto   smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto   smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ              = smem_thr_copy_Q.partition_S(sQ);

    auto   smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto   smem_thr_copy_K   = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK              = smem_thr_copy_K.partition_S(sK);

    auto   smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto   smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt             = smem_thr_copy_V.partition_S(sVt);

    // TODO: this might need to change if we change the mma instruction in SM70
    Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(acc_o)>>{});
    Tensor scores_sum = make_fragment_like(scores_max);

    // Construct identity layout for sQ and sK
    // (BLK_M, BLK_K) -> (blk_m, blk_k)
    Tensor cQ  = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    // (BLK_N, BLK_K) -> (blk_n, blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));

    // if (cute::thread0()) {
    //     print(cQ.layout()); printf("\n");
    //     for (int i = 0; i < size(cQ); ++i) {
    //         printf("%d ", get<0>(cQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(cQ); ++i) {
    //         printf("%d ", get<1>(cQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ   = gmem_thr_copy_QKV.partition_S(cQ);    // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Prologue
#if 0
  printf("Thr %d: A(%d,%d):%d  B(%d,%d):%d\n",
         threadIdx.x,
         int(get<0>(tCcA(0))), int(get<1>(tCcA(0))), int(tCpA(0)),
         int(get<0>(tCcB(0))), int(get<1>(tCcB(0))), int(tCpB(0)));
#endif

    Tensor tQrQ = make_fragment_like(tQgQ);
    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy<true>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, binfo.actual_seqlen - m_block * kBlockM);

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy<true>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, binfo.actual_seqlen - n_block * kBlockN);

    cute::cp_async_fence();

    clear(acc_o);

    for (; n_block >= 0; --n_block)
    {
        // (MMA=4, MMA_M, MMA_N)
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // NOTE: 最后一个block由于seq len不一定能整除，存在需要加载填充的情况
        const index_t cu_blockn_len = binfo.actual_seqlen - n_block * kBlockN;

        // NOTE: 提前加载V，需要有异步加载支持，T4是否有收益需要测试？
        if (n_block == n_block_max - 1)
        {
            // NOTE: 从后往前算，最后一个可能存在padding的情况
            flash::copy<true>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, cu_blockn_len);
        }
        else
        {
            tVgV.data() = tVgV.data() + (-int(kBlockN * params.qkv_row_stride));
            flash::copy<false>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV);
        }
        cute::cp_async_fence();

        flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK,
                    tiled_mma,
                    smem_tiled_copy_Q, smem_tiled_copy_K,
                    smem_thr_copy_Q, smem_thr_copy_K);
        // if (cute::thread0()) { print(acc_s); }

        // Reshape acc_s from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));

        // TODO: add mask in here
        flash::apply_mask(scores, cu_blockn_len);

        flash::cp_async_wait<0>();
        __syncthreads();
        // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tVsV); }
        // __syncthreads();

        if (n_block > 0)
        {
            // Advance gK
            tKgK.data() = tKgK.data() + (-int(kBlockN * params.qkv_row_stride));
            flash::copy<false>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // We have key_padding_mask so we'll need to Check_inf
        if (n_block == n_block_max - 1)
        {
            softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/false>(
                scores,
                scores_max,
                scores_sum,
                acc_o,
                params.scale_softmax_log2);
        }
        else
        {
            softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false>(
                scores,
                scores_max,
                scores_sum,
                acc_o,
                params.scale_softmax_log2);
        }

        // if (cute::thread0()) { print(scores_max); print(scores_sum); print(scores); }

        // Convert scores from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(scores);
        // Reshape rP from (nrow=(2, MMA_M), ncol=(2, MMA_N)) to ((2, 2, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or ((2, 2, 1), MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_rowcol_Aregs<Kernel_traits::TiledMma>(rP.layout()));

        flash::gemm_A_in_regs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }
    }

    // Epilogue

    // Reshape acc_o from (MMA=4, MMA_M, MMA_K) to (nrow=(2, MMA_M), ncol=(2, MMA_K))
    Tensor acc_o_rowcol = make_tensor(acc_o.data(), flash::convert_layout_acc_rowcol(acc_o.layout()));
    #pragma unroll
    for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi)
    {
        float sum     = scores_sum(mi);
        float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
        float scale   = inv_sum;
        #pragma unroll
        for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni)
        {
            acc_o_rowcol(mi, ni) *= scale;
        }
    }

    // if (cute::thread0()) { print(acc_o_rowcol); }

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});   // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O   = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO         = smem_thr_copy_O.retile_S(rO);      // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO         = smem_thr_copy_O.partition_D(sO);   // ((Atom,AtomNum),PIPE_M,PIPE_N)


    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t row_offset_o = binfo.o_offset(params.row_stride, bidb) +
                                    m_block * kBlockM * params.row_stride +
                                    bidh * params.head_stride;

    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto   gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO            = gmem_thr_copy_O.partition_S(sO);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO            = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));   // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);   // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)

    flash::copy<true>(gmem_tiled_copy_O, tOrO, tOgO, tOcO, binfo.actual_seqlen - m_block * kBlockM);
}

}   // namespace flash


template<typename Kernel_traits, typename Params>
inline __device__ void compute_attn(const Params &params) 
{
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout
    // pattern.

    flash::compute_attn_1rowblock<Kernel_traits>(params, bidb, bidh, m_block);
}

template<typename Kernel_traits>
__global__ void flash_fwd_kernel(flash::AttnParams params)
{
    compute_attn<Kernel_traits>(params);
}

template<typename Kernel_traits>
void run_flash_fwd_64(flash::AttnParams& params, cudaStream_t stream)
{
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);

    const int num_m_block = UP_DIV(params.s, Kernel_traits::kBlockM);

    dim3 grid(num_m_block, params.b, params.h);

    auto kernel = &flash_fwd_kernel<Kernel_traits>;
    if (smem_size >= 48 * 1024)
    {
        cudaCheck(cudaFuncSetAttribute(kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smem_size));
    }
    // int ctas_per_sm;
    // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
    // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
    printf("grid.xyz = %d, %d, %d, Kernel_traits::kNThreads: %d\n", grid.x, grid.y, grid.z, Kernel_traits::kNThreads);
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    afterKernelLaunch();
}


template<>
void run_mha_fwd_<cutlass::half_t>(flash::AttnParams &params, cudaStream_t stream) 
{
    // kHeadDim
    // kBlockM;
    // kBlockN;
    // kNWarps: equan to kBlockM / 16
    run_flash_fwd_64<flash::Kernel_Traits<64, 32, 32, 2, cutlass::half_t>>(params, stream);
}