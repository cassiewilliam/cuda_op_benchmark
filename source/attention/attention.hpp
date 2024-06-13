#pragma once

#include "operator.hpp"
#include "bytetransformer_attention/byte_attention.h"

#include <vector>

class Attention : public Operator
{
public:
    Attention(KernelType type)
        : Operator(type)
    {}

    virtual ~Attention();

    virtual ErrorCode onResize(const TensorVector& inputs,
                               const TensorVector& outputs) override;
    virtual ErrorCode onExecute(const TensorVector& inputs,
                                const TensorVector& outputs) override;
    virtual ErrorCode onFinish(const TensorVector& inputs, const TensorVector& outputs) override;

private:
    cublasHandle_t cublasHandle;
    std::shared_ptr<Tensor> buffer;

    bytetransformer::ByteAttention<bytetransformer::OperationType::HALF>* attention_layer = nullptr;
};