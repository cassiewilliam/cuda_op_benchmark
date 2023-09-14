#pragma once

#include "operator.hpp"

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
};