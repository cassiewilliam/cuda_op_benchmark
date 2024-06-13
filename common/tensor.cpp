#include "tensor.hpp"
#include "cuda_runtime.hpp"
#include "defines.hpp"

#include <cuda_runtime.h>
#include <numeric>


std::shared_ptr<Tensor> Tensor::create(const std::vector<int>& shape,
                                       MemoryType              mtype,
                                       DataType                dtype,
                                       DimFormat               format,
                                       void*                   data)
{
    return std::make_shared<Tensor>(shape, mtype, dtype, format, data);
}

Tensor::~Tensor()
{
    m_external_data = nullptr;
}

Tensor::Tensor(const std::vector<int>& shape,
               MemoryType              mtype,
               DataType                dtype,
               DimFormat               format,
               void*                   data)
{
    int data_size = 0;
    m_strides.resize(shape.size());

    if (format == NC4HW4)
    {
        data_size = shape[0] * UP_DIV(shape[1], 4) * 4 * shape[2] * shape[3];
    }
    else
    {
        data_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    }

    m_shapes = shape;
    m_format = format;
    m_dtype  = dtype;
    m_mtype  = mtype;

    if (data == nullptr)
    {
        // NOTE: 重新分配数据
        if (mtype == MemoryType::CPU)
        {
            m_cpu_data = AllocBufferWithByteSize(data_size * elemBytes());
        }
        else
        {
            m_gpu_data    = CUDARuntime::GetInstance()->getBuffer(data_size * elemBytes());
            m_is_gpu_data = true;
        }
    }
    else
    {
        m_external_data    = data;
        m_is_external_data = true;
    }

    // NOTE: currently, only support stride with elem not bytes
    //       for backup code: m_strides[shape.size() - 1] = elemBytes();
    m_strides[shape.size() - 1] = 1;
    for (int j = shape.size() - 2; j >= 0; j--)
    {
        int elem_stride = m_strides[j + 1];
        elem_stride *= shape[j + 1];
        m_strides[j] = elem_stride;
    }
}

Tensor::Tensor()
{}

Tensor::Tensor(const Tensor& rhs)
{
    m_external_data     = rhs.m_external_data;
    m_strides          = rhs.m_strides;
    m_shapes           = rhs.m_shapes;
    m_cpu_data         = rhs.m_cpu_data;
    m_format           = rhs.m_format;
    m_is_external_data = rhs.m_is_external_data;
    m_dtype            = rhs.m_dtype;
    m_is_gpu_data      = rhs.m_is_gpu_data;
    m_gpu_data         = rhs.m_gpu_data;
}

Tensor::Tensor(const Tensor&& rhs)
{
    m_external_data    = rhs.m_external_data;
    m_strides          = rhs.m_strides;
    m_shapes           = rhs.m_shapes;
    m_cpu_data         = rhs.m_cpu_data;
    m_format           = rhs.m_format;
    m_is_external_data = rhs.m_is_external_data;
    m_dtype            = rhs.m_dtype;
    m_is_gpu_data      = rhs.m_is_gpu_data;
    m_gpu_data         = rhs.m_gpu_data;
}

Tensor& Tensor::operator=(const Tensor& rhs)
{
    m_external_data    = rhs.m_external_data;
    m_strides          = rhs.m_strides;
    m_shapes           = rhs.m_shapes;
    m_cpu_data         = rhs.m_cpu_data;
    m_format           = rhs.m_format;
    m_is_external_data = rhs.m_is_external_data;
    m_dtype            = rhs.m_dtype;
    m_is_gpu_data      = rhs.m_is_gpu_data;
    m_gpu_data         = rhs.m_gpu_data;
    return *this;
}

Tensor& Tensor::operator=(const Tensor&& rhs)
{
    m_external_data    = rhs.m_external_data;
    m_strides          = rhs.m_strides;
    m_shapes           = rhs.m_shapes;
    m_cpu_data         = rhs.m_cpu_data;
    m_format           = rhs.m_format;
    m_is_external_data = rhs.m_is_external_data;
    m_dtype            = rhs.m_dtype;
    m_is_gpu_data      = rhs.m_is_gpu_data;
    m_gpu_data         = rhs.m_gpu_data;
    return *this;
}

const std::vector<int>& Tensor::shapes() const
{
    return m_shapes;
}

const std::vector<int>& Tensor::strides() const
{
    return m_strides;
}

void* Tensor::data() const
{
    // NOTE: 优先GPU数据获取
    return m_is_gpu_data ? m_gpu_data.get() : m_cpu_data.get();
}

void* Tensor::data()
{
    // NOTE: 优先GPU数据获取
    return m_is_gpu_data ? m_gpu_data.get() : m_cpu_data.get();
}

int Tensor::size() const
{
    return std::accumulate(m_shapes.begin(), m_shapes.end(), 1, std::multiplies<int>());
}

DimFormat Tensor::format() const
{
    return m_format;
}

int Tensor::batch() const
{
    return m_shapes[0];
}

int Tensor::channel() const
{
    if (m_format == NCHW)
    {
        return m_shapes[1];
    }
    else if (m_format == NHWC)
    {
        return m_shapes[3];
    }
    else if (m_format == NC4HW4)
    {
        return m_shapes[1];
    }

    return -1;
}

int Tensor::width() const
{
    if (m_format == NCHW)
    {
        return m_shapes[3];
    }
    else if (m_format == NHWC)
    {
        return m_shapes[2];
    }
    else if (m_format == NC4HW4)
    {
        return m_shapes[3];
    }

    return -1;
}

int Tensor::height() const
{
    if (m_format == NCHW)
    {
        return m_shapes[2];
    }
    else if (m_format == NHWC)
    {
        return m_shapes[1];
    }
    else if (m_format == NC4HW4)
    {
        return m_shapes[2];
    }

    return -1;
}

void Tensor::fillData(bool random) const
{
    if (m_mtype == MemoryType::CPU)
    {
        if (m_dtype == FLOAT32)
        {
            for (size_t j = 0; j < size(); ++j)
            {
                ((float*)m_cpu_data.get())[j] = rand() % 100 / 100.0f;
            }
        }
        else if (m_dtype == FLOAT16)
        {
            if (random)
            {
                for (size_t j = 0; j < size(); ++j)
                {
                    auto value = (rand() % 255 - 128) / 256.0f;
                    ((half *)m_cpu_data.get())[j] = half(value);
                }
            }
            else
            {
                // NOTE: for debug
                for (int i = 0; i < m_shapes[0]; ++i)
                {
                    for (int j = 0; j < m_shapes[1]; ++j)
                    {
                        for (int k = 0; k < m_shapes[2]; ++k)
                        {
                            for (int m = 0; m < m_shapes[3]; ++m)
                            {
                                int offset = i * m_strides[0] + j * m_strides[1] + k * m_strides[2] + m * m_strides[3];
                                ((half *)m_cpu_data.get())[offset] = half(float(offset % 65535));
                            }
                        }
                    }
                }
            }
        }
        else if (m_dtype == UINT8)
        {
            for (size_t j = 0; j < size(); ++j)
            {
                ((uint8_t*)m_cpu_data.get())[j] = rand() % 255;
            }
        }
        else if (m_dtype == INT8)
        {
            for (size_t j = 0; j < size(); ++j)
            {
                ((int8_t*)m_cpu_data.get())[j] = rand() % 255 - 128;
            }
        }
    }
    else if (m_mtype == MemoryType::GPU)
    {
        if (m_dtype == DataType::FLOAT32)
        {
            initialize_matrix<float>((float*)m_gpu_data.get(),
                                     shape(0) * shape(1) * shape(2),
                                     shape(3),
                                     1);
        }
        else if (m_dtype == DataType::FLOAT16)
        {
            initialize_matrix<half>((half*)m_gpu_data.get(),
                                     shape(0) * shape(1) * shape(2),
                                     shape(3),
                                     1);
        }
        else if (m_dtype == DataType::INT8)
        {
            initialize_matrix<int8_t>((int8_t*)m_gpu_data.get(),
                                      shape(0) * shape(1) * shape(2),
                                      shape(3),
                                      1);
        }
    }
}

bool Tensor::isCUDA() const
{
    return m_is_gpu_data;
}

bool Tensor::isCPU() const
{
    return !m_is_gpu_data;
}

void Tensor::toCPU()
{
    if (m_is_gpu_data)
    {
        m_cpu_data = AllocBufferWithByteSize(size() * elemBytes());

        CUDARuntime::GetInstance()->memcpy(m_cpu_data.get(),
                                           m_gpu_data.get(),
                                           size() * elemBytes(),
                                           MemcpyDeviceToHost);

        m_is_gpu_data = false;
        m_mtype = MemoryType::CPU;
    }
}

void Tensor::toCUDA()
{
    if (m_gpu_data)
    {
        LOGE("already memory in cuda\n");
        return;
    }

    if (!m_is_gpu_data)
    {
        auto runtime = CUDARuntime::GetInstance();

        m_gpu_data = runtime->getBuffer(size() * elemBytes());

        runtime->memcpy(m_gpu_data.get(),
                        m_cpu_data.get(),
                        size() * elemBytes(),
                        MemcpyHostToDevice);

        m_is_gpu_data = true;
        m_mtype = MemoryType::GPU;
    }
}

int Tensor::elemBytes() const
{
    switch (m_dtype)
    {
    case INT32:
        return 4;
    case UINT32:
        return 4;
    case FLOAT32:
        return 4;
    case BFLOAT16:
        return 2;
    case FLOAT16:
        return 2;
    case INT8:
        return 1;
    case UINT8:
        return 1;
    default:
        LOGE("not support type: %d", m_dtype);
        break;
    }
    return -1;
}

int Tensor::shape(int index) const
{
    int size = m_shapes.size();
    return m_shapes[(index + size) % size];
}

int Tensor::stride(int index) const
{
    int size = m_strides.size();
    return m_strides[(index + size) % size];
}

DataType Tensor::dtype() const
{
    return m_dtype;
}

MemoryType Tensor::mtype() const
{
    return m_mtype;
}

int Tensor::seqlen() const
{
    return m_shapes[1];
}

int Tensor::numheads() const
{
    return m_shapes[2];
}

int Tensor::headdim() const
{
    return m_shapes[3];
}

void* Tensor::cpu_data() const
{
    return m_cpu_data.get();
}

void* Tensor::cpu_data()
{
    return m_cpu_data.get();
}