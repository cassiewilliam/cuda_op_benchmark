#pragma once

#include "enum.hpp"
#include <vector_types.h>
#include "cutlass/half.h"

#include <memory>
#include <string>
#include <vector>

/**
 * @brief 支持CPU和GPU数据的Tensor
 *        假定只能支持一种类型的内存数据，优先顺序为SMEM>GPU>CPU
 */
class Tensor
{
public:
    static std::shared_ptr<Tensor> create(const std::vector<int>& shape,
                                          MemoryType              mtype  = MemoryType::CPU,
                                          DataType                dtype  = DataType::FLOAT32,
                                          DimFormat               format = DimFormat::NCHW,
                                          void*                   data   = nullptr);

    // shape - the shape of the tensor, with size n
    // data - the buffer of the tensor, must not be null with size equals
    //        shape[0] * shape[1] * ... * shape[n-1]
    Tensor(const std::vector<int>& shape,
           MemoryType              mtype  = MemoryType::CPU,
           DataType                dtype  = DataType::FLOAT32,
           DimFormat               format = DimFormat::NCHW,
           void*                   data   = nullptr);

    Tensor();
    Tensor(const Tensor &other);
    Tensor(const Tensor &&other);
    Tensor &operator=(const Tensor &other);
    Tensor &operator=(const Tensor &&other);
    ~Tensor();

    const std::vector<int>& shapes() const;
    const std::vector<int>& strides() const;

    int shape(int index) const;
    int stride(int index) const;

    void* data() const;
    void* data();

    void* cpu_data() const;
    void* cpu_data();

    DimFormat  format() const;
    DataType   dtype() const;
    MemoryType mtype() const;

    int size() const;
    int batch() const;

    int channel() const;
    int width() const;
    int height() const;

    int seqlen() const;
    int numheads() const;
    int headdim() const;
    int elemBytes() const;

    void fillData(bool random = true) const;

    bool isCUDA() const;
    bool isCPU() const;

    void toCPU();
    void toCUDA();

private:
    std::vector<int>         m_shapes;
    std::vector<int>         m_strides;

    void* m_external_data = nullptr;

    std::shared_ptr<uint8_t> m_cpu_data;   // NOTE: cpu data pointer
    std::shared_ptr<uint8_t> m_gpu_data;   // NOTE: gpu data pointer

    DimFormat  m_format = DimFormat::NCHW;
    DataType   m_dtype  = DataType::FLOAT32;
    MemoryType m_mtype  = MemoryType::CPU;

    bool m_is_external_data = false;   // NOTE: 外部数据
    bool m_is_gpu_data      = false;   // NOTE: GPU内存
};