#include "gemm.hpp"
#include "gemm_runner.hpp"

#include <cuda_runtime.h>

GemmOp::~GemmOp()
{
    if (m_allocator)
    {
        delete m_allocator;
        m_allocator = nullptr;
    }

    if (m_cublas_wrapper)
    {
        delete m_cublas_wrapper;
        m_cublas_wrapper = nullptr;
    }

    if (cublas_algo_map)
    {
        delete cublas_algo_map;
        cublas_algo_map = nullptr;
    }

    if (cublas_wrapper_mutex)
    {
        delete cublas_wrapper_mutex;
        cublas_wrapper_mutex = nullptr;
    }
}

ErrorCode GemmOp::onResize(const TensorVector &inputs,
                           const TensorVector &outputs)
{
    auto runtime = CUDARuntime::GetInstance();

    cudaStream_t stream = runtime->getCurrentCUDAStream();

    cublasCreate(&m_cublas_handle);
    cublasLtCreate(&m_cublaslt_handle);
    cublasSetStream(m_cublas_handle, stream);

    cublas_algo_map = new ft::cublasAlgoMap("./gemm_config.in");

    m_allocator = new ft::Allocator<ft::AllocatorType::CUDA>(runtime->getDevice());
    cublas_wrapper_mutex = new std::mutex();

    m_cublas_wrapper = new ft::cublasMMWrapper(m_cublas_handle, m_cublaslt_handle, stream, cublas_algo_map, cublas_wrapper_mutex, m_allocator);
    m_cublas_wrapper->setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F);

    return NO_ERROR;
}

ErrorCode GemmOp::onExecute(const TensorVector& inputs,
                            const TensorVector& outputs)
{
    auto runtime = CUDARuntime::GetInstance();

    auto &A = inputs[0];
    auto &B = inputs[1];
    auto &C = inputs[2];
    auto &Bias = inputs[3];
    auto &P = inputs[4];

    auto &D = outputs[0];

    const int m = A->shape(-2);
    const int k = A->shape(-1);
    const int n = B->shape(-2);

    float alpha = ((float *)P->data())[0], beta = ((float *)P->data())[1];

    if (m_type == KernelType::KT_HGEMM_Naive || 
        m_type == KernelType::KT_HGEMM_WMMA ||
        m_type == KernelType::KT_HGEMM_MMA_PTX ||
        m_type == KernelType::KT_HGEMM_MMA_PTX_OPT ||
        m_type == KernelType::KT_HGEMM_CUTLASS)
    {
        gemm_runner(false, false, m, n, k, &alpha, A->data(), A->dtype(), B->data(), B->dtype(), &beta, C->data(), C->dtype(), D->data(), D->dtype(), m_type);
    }
    else if (m_type == KernelType::KT_HGEMM_BERT) 
    {
        const __half halpha = (__half)(alpha), hbeta = (__half)(beta);
        cublasGemmEx(m_cublas_handle,
                     CUBLAS_OP_N,
                     CUBLAS_OP_N,
                     n, m, k,
                     &halpha,
                     B->data(), CUDA_R_16F, n,
                     A->data(), CUDA_R_16F, k,
                     &hbeta,
                     D->data(), CUDA_R_16F, n,
                     CUDA_R_16F,
                     static_cast<cublasGemmAlgo_t>(99));
    }
    else if (m_type == KT_HGEMM_ADDBIAS)
    {
        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
        // cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        m_cublas_wrapper->GemmX(CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                n, m, k,
                                B->data(), n,
                                A->data(), k,
                                D->data(), n,
                                Bias->data(),
                                epilogue,
                                alpha, beta);
    }
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