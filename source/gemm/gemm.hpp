#pragma once

#include "operator.hpp"
#include "cublasMMWrapper.h"
#include "allocator.h"
#include "cublasAlgoMap.h"

#include <vector>

class GemmOp : public Operator
{
public:
    GemmOp(KernelType type)
        : Operator(type)
    {}

    virtual ~GemmOp();

    virtual ErrorCode onResize(const TensorVector& inputs, const TensorVector& outputs) override;
    virtual ErrorCode onExecute(const TensorVector& inputs, const TensorVector& outputs) override;
    virtual ErrorCode onFinish(const TensorVector& inputs, const TensorVector& outputs) override;
private:
    ft::Allocator<ft::AllocatorType::CUDA>* m_allocator = nullptr;
    ft::cublasMMWrapper*                m_cublas_wrapper = nullptr;

    std::mutex*        cublas_wrapper_mutex = nullptr;
    ft::cublasAlgoMap* cublas_algo_map = nullptr;
    cublasHandle_t     m_cublas_handle;
    cublasLtHandle_t   m_cublaslt_handle;
};