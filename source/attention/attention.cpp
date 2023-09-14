
#include "attention.hpp"
#include "flash_attention/flash_attention.hpp"
#include "unfused_attention/unfused_attention.hpp"

#include <cuda_runtime.h>

Attention::~Attention()
{
    // Do nothing
}

ErrorCode Attention::onResize(const TensorVector& inputs, const TensorVector& outputs)
{
    auto runtime = CUDARuntime::GetInstance();
    int  device;

    cudaGetDevice(&device);
    int       max_smem_per_sm, max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(&max_smem_per_sm,
                                               cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                                               device);
    status_           = cudaDeviceGetAttribute(&max_smem_per_block,
                                     cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                     device);
    printf("max_smem_per_sm = %d, max_smem_per_block = %d\n", max_smem_per_sm, max_smem_per_block);

    return NO_ERROR;
}

ErrorCode Attention::onExecute(const TensorVector& inputs, const TensorVector& outputs)
{
    auto runtime = CUDARuntime::GetInstance();

    auto& q = inputs[0];
    auto& k = inputs[1];
    auto& v = inputs[2];

    auto& o = outputs[0];

    if (m_type == KernelType::KT_UnFusedAttention)
    {
        // unfused_attention(*q, *k, *v, *o);
    }
    else if (m_type == KernelType::KT_FlashAttention)
    {
        // auto result = flash_attention1(*q, *k, *v, *o, 0.0f, 0.05f, false, false);
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