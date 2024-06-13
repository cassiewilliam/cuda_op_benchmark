#include "attention.hpp"
#include "handle_fa_wo_cutlass/flash_attention.hpp"
#include "unfused_attention/unfused_attention.hpp"
#include "flash_attention_for_t4/flash_byte_attention.hpp"
#include "unfused_attention_cublas/ft_bert_attention.hpp"
#include "flash_attention2/flash_attention2.hpp"

#include <cuda_runtime.h>

Attention::~Attention()
{
    // Do nothing
}

ErrorCode Attention::onResize(const TensorVector& inputs, const TensorVector& outputs)
{
    cublasCreate(&cublasHandle);

    cublasSetStream(cublasHandle, CUDARuntime::GetInstance()->getCurrentCUDAStream());

    auto qkv_shape = inputs[0]->shapes();
    auto batch_size = qkv_shape[0];
    auto seq_len = qkv_shape[1];
    auto head_num = qkv_shape[2];
    auto size_per_head = qkv_shape[3];

    int buf_size = batch_size * head_num * seq_len * size_per_head;
    int qk_buf_size = batch_size * head_num * seq_len * seq_len;

    buffer = Tensor::create({1, 1, 1, buf_size * 4 + qk_buf_size}, MemoryType::GPU, DataType::FLOAT16);

    // if (m_type == KernelType::KT_ByteAttention)
    // {
    //     attention_layer = new ByteAttention<bytetransformer::OperationType::HALF>(batch_size, head_num, size_per_head, seq_len, true, true); 
    // }

    return NO_ERROR;
}

ErrorCode Attention::onExecute(const TensorVector& inputs, const TensorVector& outputs)
{
    auto runtime = CUDARuntime::GetInstance();

    auto& q = inputs[0];
    auto& k = inputs[1];
    auto& v = inputs[2];
    auto& qkv = inputs[3];
    auto& mask = inputs[4];
    auto& seq_lens = inputs[5];

    constexpr int MAX_SEQ_LEN = 256;

    auto& out = outputs[0];

    if (m_type == KernelType::KT_UnFusedAttention)
    {
        // unfused_attention(*q, *k, *v, *out);
    }
    else if (m_type == KernelType::KT_HandleFlashAttentionWithoutCutlass)
    {
        // auto result = flash_attention1(*q, *k, *v, *o, 0.0f, 0.05f, false, false);
    }
    else if (m_type == KernelType::KT_FlashAttention2)
    {
        // flash_attention2(*q, *k, *v, *o, 1.0f, true);
        flash_attention2(*qkv, *q, *k, *v, *mask, *out, *seq_lens, MAX_SEQ_LEN);
    }
    else if (m_type == KernelType::KT_FlashAttention_For_T4)
    {
        // flash_byte_attention(*qkv, *qkv, *mask, *out, *seq_lens, MAX_SEQ_LEN);
        flash_byte_attention(*qkv, *q, *k, *v, *mask, *out, *seq_lens, MAX_SEQ_LEN);
    }
    else if (m_type == KernelType::KT_ByteTransformerAttention)
    {
        auto stream = CUDARuntime::GetInstance()->getCurrentCUDAStream();
        ft_bert_attention(*q, *k, *v, *q, *k, *v, *mask, *out, *buffer, stream, cublasHandle);
    }
    else
    {
        return NOT_SUPPORT;
    }

    return NO_ERROR;
}

ErrorCode Attention::onFinish(const TensorVector& inputs, const TensorVector& outputs)
{
    return NO_ERROR;
}

class AttentionCreator : public Creator
{
public:
    virtual Operator* onCreate(KernelType type) const override
    {
        return new Attention(type);
    }
};

static CreatorRegister<AttentionCreator> __init(OpType_Attention);
