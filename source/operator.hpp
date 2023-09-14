#pragma once

#include "tensor.hpp"
#include "cuda_runtime.hpp"
#include "creator_factory.hpp"

#include <map>

class Operator
{
public:
    using TensorVector = std::vector<std::shared_ptr<Tensor>>;

    /**
     * @brief initializer.
     * @param backend   backend that exection will running on.
     */
    Operator(KernelType type)
        : m_type(type)
    {}

    /**
     * @brief deinitializer.
     */
    virtual ~Operator() = default;

    /**
     * @brief response shape change of input or output tensors.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return resize result
     */
    virtual ErrorCode onResize(const TensorVector& inputs, const TensorVector& outputs)
    {
        return NO_ERROR;
    }

    /**
     * @brief perform execution.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return execution result
     */
    virtual ErrorCode onExecute(const TensorVector& inputs, const TensorVector& outputs) = 0;

    /**
     * @brief perform finish.
     * @param inputs    input tensors
     * @param outputs   output tensors
     * @return execution result
     */
    virtual ErrorCode onFinish(const TensorVector& inputs, const TensorVector& outputs) = 0;

protected:
    KernelType m_type;
};