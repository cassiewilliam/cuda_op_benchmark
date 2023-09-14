#include "gemm.hpp"
#include "gemm_runner.hpp"

#include <cuda_runtime.h>

GemmOp::~GemmOp()
{
    // Do nothing
}

ErrorCode GemmOp::onResize(const TensorVector &inputs,
                           const TensorVector &outputs)
{
    auto runtime = CUDARuntime::GetInstance();

    int           device;
    cudaGetDevice(&device);
    int       max_smem_per_sm, max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(&max_smem_per_sm,
                                               cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                                               device);
    status_           = cudaDeviceGetAttribute(&max_smem_per_block,
                                     cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                     device);
    printf("max_smem_per_sm = %d, max_smem_per_block = %d\n",
           max_smem_per_sm / 1024, max_smem_per_block / 1024);

    return NO_ERROR;
}

ErrorCode GemmOp::onExecute(const TensorVector& inputs,
                            const TensorVector& outputs)
{
    auto runtime = CUDARuntime::GetInstance();

    auto &A = inputs[0];
    auto &B = inputs[1];
    auto &C = inputs[2];
    auto &P = inputs[3];

    auto &D = outputs[0];

    const int m = A->shape(-2);
    const int k = A->shape(-1);
    const int n = B->shape(-2);

    float alpha = ((float *)P->data())[0], beta = ((float *)P->data())[1];
    gemm_runner(false, false, m, n, k, &alpha, A->data(), A->dtype(), B->data(), B->dtype(), &beta, C->data(), C->dtype(), D->data(), D->dtype(), m_type);

    return NO_ERROR;
}

ErrorCode GemmOp::onFinish(const TensorVector& inputs,
                           const TensorVector& outputs)
{
    return ErrorCode::NO_ERROR;
}

class GemmCreator : public Creator
{
public:
    virtual Operator* onCreate(KernelType type) const override
    {
        return new GemmOp(type);
    }
};

static CreatorRegister<GemmCreator> __init(OpType_Gemm);