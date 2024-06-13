/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch
// headers.

#include "flash_decode_attention.hpp"

#include "cuda_runtime.hpp"
#include "flash_attn_kernel.cuh"

#include <cutlass/numeric_types.h>

constexpr int kHeadDim = 128;

void set_params_fprop(Flash_fwd_params& params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const Tensor q,
                      const Tensor k,
                      const Tensor v,
                      Tensor       out,
                      void*        cu_seqlens_q_d,
                      void*        cu_seqlens_k_d,
                      void*        p_d,
                      void*        softmax_lse_d,
                      float        p_dropout,
                      float        softmax_scale,
                      int          window_size_left,
                      int          window_size_right)
{

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.is_bf16 = false;   // q.dtype() == torch::kBFloat16;

    // Set the pointers and strides.
    params.q_ptr = q.data();
    params.k_ptr = k.data();
    params.v_ptr = v.data();
    // All stride are in elements, not bytes.
    params.q_row_stride  = q.stride(-3);
    params.k_row_stride  = k.stride(-3);
    params.v_row_stride  = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr         = out.data();
    params.o_row_stride  = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_q_d == nullptr)
    {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
    }

    params.cu_seqlens_q = static_cast<int*>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int*>(cu_seqlens_k_d);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b                = b;
    params.h                = h;
    params.h_k              = h_k;
    params.h_h_k_ratio      = h / h_k;
    params.seqlen_q         = seqlen_q;
    params.seqlen_k         = seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d                = d;
    params.d_rounded        = d_rounded;

    // Set the different scale values.
    params.scale_softmax      = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t     = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout               = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    // TORCH_CHECK(p_dropout < 1.f);

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;

    if (window_size_left < 0 && window_size_right >= 0)
    {
        window_size_left = seqlen_k;
    }
    if (window_size_left >= 0 && window_size_right < 0)
    {
        window_size_right = seqlen_k;
    }
    params.window_size_left  = window_size_left;
    params.window_size_right = window_size_right;

    params.is_seqlens_k_cumulative = true;
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
inline int num_splits_heuristic(int batch_nheads_mblocks,
                                int num_SMs,
                                int num_n_blocks,
                                int max_splits)
{
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs)
    {
        return 1;
    }
    max_splits                        = std::min({max_splits, num_SMs, num_n_blocks});
    float              max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 ||
               ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        if (!is_split_eligible(num_splits))
        {
            efficiency.push_back(0.f);
        }
        else
        {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff     = n_waves / ceil(n_waves);
            // printf("num_splits = %d, eff = %f\n", num_splits, eff);
            if (eff > max_efficiency)
            {
                max_efficiency = eff;
            }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++)
    {
        if (!is_split_eligible(num_splits))
        {
            continue;
        }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency)
        {
            return num_splits;
        }
    }
    return 1;
}

void run_mha_fwd(Flash_fwd_params& params, cudaStream_t stream, bool force_split_kernel = false)
{
    if (params.num_splits <= 1 && !force_split_kernel)
    {   // If we don't set it num_splits == 0
        run_mha_fwd_<cutlass::half_t, kHeadDim>(params, stream);
    }
    else
    {
        run_mha_fwd_splitkv_dispatch<cutlass::half_t, kHeadDim>(params, stream);
    }
}

void flash_decode_attention(Tensor&       q,     // batch_size x seqlen_q x num_heads x head_size
                            const Tensor& k,     // batch_size x seqlen_k x num_heads_k x head_size
                            const Tensor& v,     // batch_size x seqlen_k x num_heads_k x head_size
                            Tensor&       out,   // batch_size x seqlen_q x num_heads x head_size
                            const float   softmax_scale,
                            bool          is_causal)
{
    // FlashAttention only supports Ampere GPUs or newer. && FlashAttention only supports Turing GPUs or newer.
    // FlashAttention only support fp16 and bf16 data type, bfloat16 is only supported on Ampere GPUs or newer
    // Input tensor must have contiguous last dimension 输入的qkv必须是连续的
    // FlashAttention forward only supports head dimension at most 256
    // Number of heads in key/value must divide number of heads in query
    auto dprops = CUDARuntime::GetInstance()->prop();

    // Local window size
    int window_size_left=0, window_size_right=0;
    float p_dropout = 0.0;
    printf("WRONG Window size\n");
    printf("WRONG p_dropout\n");

    // bool is_sm75 = dprops->major == 7 && dprops->minor == 5;
    bool is_sm8x = dprops->major == 8 && dprops->minor >= 0;
    bool is_sm90 = dprops->major == 9 && dprops->minor == 0;
 
    const auto sizes = q.shapes();

    const int batch_size   = sizes[0];
    int       seqlen_q     = sizes[1];
    int       num_heads    = sizes[2];
    const int head_size_og = sizes[3];
    const int seqlen_k     = k.shape(1);
    const int num_heads_k  = k.shape(2);
 
    printf("q info: batch_size=%d, seqlen_q=%d, num_heads=%d, head_size_og=%d\n", batch_size, seqlen_q, num_heads, head_size_og);
    printf("k info: seqlen_k=%d, num_heads_k=%d\n", seqlen_k, num_heads_k);

    if (seqlen_q == 1)
    {
        is_causal = false;
    }   // causal=true is the same as causal=false in this case
    if (is_causal)
    {
        window_size_right = 0;
    }

    // Faster to transpose q from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d) in
    // this case H/t Daniel Haziza
    const int seqlenq_ngroups_swapped = seqlen_q == 1 && num_heads > num_heads_k &&
                                        window_size_left < 0 && window_size_right < 0 &&
                                        p_dropout == 0.f && head_size_og % 8 == 0;
    printf("Wrong param seqlenq_ngroups_swapped: %d\n", seqlenq_ngroups_swapped);
    if (seqlenq_ngroups_swapped)
    {
        printf("WRONG\n");
        /*
        const int ngroups = num_heads / num_heads_k;
        q         = q.reshape({batch_size, num_heads_k, ngroups, head_size_og}).transpose(1, 2);
        seqlen_q  = ngroups;
        num_heads = num_heads_k;
        */
    }

    //CHECK_SHAPE(q, batch_size, seqlen_q, num_heads, head_size_og);
    //CHECK_SHAPE(k, batch_size, seqlen_k, num_heads_k, head_size_og);
    //CHECK_SHAPE(v, batch_size, seqlen_k, num_heads_k, head_size_og);

    Tensor q_padded, k_padded, v_padded;
    if (head_size_og % 8 != 0)
    {
        printf("WRONG NO PAD NO PAD!\n");
        /*
        q_padded = torch::nn::functional::pad(
            q,
            torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        k_padded = torch::nn::functional::pad(
            k,
            torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
        v_padded = torch::nn::functional::pad(
            v,
            torch::nn::functional::PadFuncOptions({0, 8 - head_size_og % 8}));
            */
    }
    else
    {
        q_padded = q;
        k_padded = k;
        v_padded = v;
    }

    printf("NO Need out\n");
    /*
    Tensor out;
    if (out_.has_value())
    {
        out = out_.value();
        //TORCH_CHECK(out.dtype() == q_dtype, "Output must have the same dtype as inputs");
        //CHECK_DEVICE(out);
        //TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        //CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_og);
        if (head_size_og % 8 != 0)
        {
            out = torch::empty_like(q_padded);
        }
    }
    else
    {
        out = torch::empty_like(q_padded);
    }
    */

    auto      round_multiple    = [](int x, int m) { return (x + m - 1) / m * m; };
    const int head_size         = round_multiple(head_size_og, 8);
    const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded  = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded  = round_multiple(seqlen_k, 128);

    // Otherwise the kernel will be launched from cuda:0 device
    // Cast to char to avoid compiler warning about narrowing
    //at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    //auto opts = q.options();
    //auto   softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));

    auto softmax_lse = Tensor::create(
                {batch_size, num_heads, seqlen_q},
                MemoryType::GPU,
                FLOAT32);

    Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    
    bool return_softmax = false;
    if (return_softmax)
    {
        //TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        //p = torch::empty({batch_size, num_heads, seqlen_q_rounded, seqlen_k_rounded}, opts);
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q,
                     seqlen_k,
                     seqlen_q_rounded,
                     seqlen_k_rounded,
                     num_heads,
                     num_heads_k,
                     head_size,
                     head_size_rounded,
                     q_padded,
                     k_padded,
                     v_padded,
                     out,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     return_softmax ? p.data() : nullptr,
                     softmax_lse->data(),
                     p_dropout,
                     softmax_scale,
                     window_size_left,
                     window_size_right);

    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n      = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (seqlen_k + block_n - 1) / block_n;
    // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
    // In any case we don't expect seqlen_q to be larger than 64 for inference.
    const int num_m_blocks = (seqlen_q + 64 - 1) / 64;
    params.num_splits      = 1;
    if (p_dropout == 0.0f)
    {   // SplitKV is not implemented for dropout
        params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks,
                                                 dprops->multiProcessorCount,
                                                 num_n_blocks,
                                                 128);
        if (params.num_splits > 1)
        {
            // Tensor softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads,
            // seqlen_q}, opts.dtype(at::kFloat)); Tensor out_accum =
            // torch::empty({params.num_splits, batch_size, num_heads, seqlen_q, head_size_rounded},
            // opts.dtype(at::kFloat));

            auto softmax_lse_accum = Tensor::create(
                {params.num_splits, batch_size, num_heads, seqlen_q},
                MemoryType::GPU,
                FLOAT32);
            auto out_accum = Tensor::create(
                {params.num_splits, batch_size, num_heads, seqlen_q, head_size_rounded},
                MemoryType::GPU,
                FLOAT32);

            params.softmax_lseaccum_ptr = softmax_lse_accum->data();
            params.oaccum_ptr           = out_accum->data();
        }
        //TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }

    // number of times random will be generated per thread, to offset philox counter in thc random
    // state
    // We use a custom RNG that increases the offset by batch_size * nheads * 32.
    int64_t counter_offset = params.b * params.h * 32;

    // auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));

    auto rng_state = Tensor::create({2}, MemoryType::GPU, INT64);
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state->data());

    if (p_dropout > 0.0)
    {
        /*
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            gen_,
            at::cuda::detail::getDefaultCUDAGenerator());
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen->mutex_);
        params.philox_args = gen->philox_cuda_state(counter_offset);
        */
    }

    auto stream = CUDARuntime::GetInstance()->getCurrentCUDAStream();
    run_mha_fwd(params, stream);

    /*
    Tensor out_padded = out;
    if (head_size_og % 8 != 0)
    {
        out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
        if (out_.has_value())
        {
            out_.value().copy_(out);
        }
    }

    if (seqlenq_ngroups_swapped)
    {
        out = out.transpose(1, 2).reshape({batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        out_padded = out_padded.transpose(1, 2).reshape(
            {batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        q_padded = q_padded.transpose(1, 2).reshape(
            {batch_size, 1, num_heads_k * seqlen_q, head_size_og});
        softmax_lse = softmax_lse.reshape({batch_size, num_heads_k * seqlen_q, 1});
    }

    // return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p, rng_state};
    */
}