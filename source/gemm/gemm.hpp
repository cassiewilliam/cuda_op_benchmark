#pragma once

#include "operator.hpp"

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
};