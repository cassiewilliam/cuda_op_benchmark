#include "cuda_runtime.hpp"

CUDARuntime* CUDARuntime::GetInstance(int device_id)
{
    static CUDARuntime*   s_instance = nullptr;
    static std::once_flag s_instance_flag;
    std::call_once(s_instance_flag,
                   [device_id]() { s_instance = new (std::nothrow) CUDARuntime(device_id); });
    return s_instance;
}

CUDARuntime::CUDARuntime(int device_id)
{
    int version;
    cudaCheck(cudaRuntimeGetVersion(&version));
    int id = device_id;
    if (id < 0)
        cudaCheck(cudaGetDevice(&id));

    LOGD("start CUDARuntime id:%d, version:%d\n", id, version);

    cudaCheck(cudaSetDevice(id));

    m_device_id = id;
    cudaCheck(cudaGetDeviceProperties(&m_properties, id));
    assert((m_properties.maxThreadsPerBlock > 0) && "create runtime error");

    assert(0 == cudaStreamCreate(&m_stream) && "create stream error");
}

CUDARuntime::~CUDARuntime()
{
    cudaStreamDestroy(m_stream);
    LOGD("end ~CUDARuntime !\n");
}

std::shared_ptr<uint8_t> CUDARuntime::getBuffer(size_t size_in_bytes)
{
    return std::shared_ptr<uint8_t>((uint8_t*)this->alloc(size_in_bytes),
                                    [this](void* p) { this->free(p); });
}

void* CUDARuntime::alloc(size_t size_in_bytes)
{
    void* ptr = nullptr;
    cudaCheck(cudaMalloc(&ptr, size_in_bytes));
    assert((nullptr != ptr) && "alloc cuda memory failed");

    checkKernelErrors();
    return ptr;
}

void CUDARuntime::free(void* ptr)
{
    cudaCheck(cudaFree(ptr));
    checkKernelErrors();
}

void CUDARuntime::memcpy(void*         dst,
                         const void*   src,
                         size_t        size_in_bytes,
                         MemCopyType_t type,
                         bool          sync)
{
    cudaMemcpyKind cuda_kind;
    switch (type)
    {
    case MemcpyDeviceToHost:
        cuda_kind = cudaMemcpyDeviceToHost;
        break;
    case MemcpyHostToDevice:
        cuda_kind = cudaMemcpyHostToDevice;
        break;
    case MemcpyDeviceToDevice:
        cuda_kind = cudaMemcpyDeviceToDevice;
        break;
    default:
        printf("bad cuda memcpy kind\n");
    }

    cudaCheck(cudaMemcpy(dst, src, size_in_bytes, cuda_kind));
    checkKernelErrors();
}

void CUDARuntime::memset(void* dst, int value, size_t size_in_bytes)
{
    cudaCheck(cudaMemset(dst, value, size_in_bytes));
    checkKernelErrors();
}

size_t CUDARuntime::blocksNum(const size_t total_threads)
{
    return (total_threads + m_thread_per_block - 1) / m_thread_per_block;
}

int CUDARuntime::getDevice()
{
    int current_dev_id = 0;
    cudaCheck(cudaGetDevice(&current_dev_id));
    return current_dev_id;
}