#pragma once

#include "enum.hpp"

#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <sstream>
#include <string>
#include <vector>

typedef enum
{
    CUDA_FLOAT32 = 0,
    CUDA_FLOAT16 = 1,
} CUDADataType_t;

typedef enum
{
    MemcpyHostToDevice   = 1,
    MemcpyDeviceToHost   = 2,
    MemcpyDeviceToDevice = 3,
} MemCopyType_t;

#define cudaCheck(_x)                                       \
    do                                                      \
    {                                                       \
        cudaError_t _err = (_x);                            \
        if (_err != cudaSuccess)                            \
        {                                                   \
            printf("Check failed: %d ==> %s\n", _err, #_x); \
        }                                                   \
    } while (0)

#define afterKernelLaunch()            \
    do                                 \
    {                                  \
        cudaCheck(cudaGetLastError()); \
    } while (0)

#ifdef DEBUG
#    define checkKernelErrors()                         \
        do                                              \
        {                                               \
            cudaDeviceSynchronize();                    \
            cudaError_t __err = cudaGetLastError();     \
            if (__err != cudaSuccess)                   \
            {                                           \
                printf("File:%s Line %d: failed: %s\n", \
                       __FILE__,                        \
                       __LINE__,                        \
                       cudaGetErrorString(__err));      \
                abort();                                \
            }                                           \
        } while (0)

#else
#    define checkKernelErrors() ((void)0)
#endif

/// Simple function to initialize a matrix to arbitrary small elements.
template<typename T>
cudaError_t initialize_matrix(T* matrix, int rows, int columns, int seed = 0);

class CUDARuntime
{
public:
    static CUDARuntime* GetInstance(int device_id = -1);

public:
    CUDARuntime(int device_id);
    ~CUDARuntime();
    CUDARuntime(const CUDARuntime&)            = delete;
    CUDARuntime& operator=(const CUDARuntime&) = delete;

    int deviceId() const;

    void activate();

    std::shared_ptr<uint8_t> getBuffer(size_t size_in_bytes);

    void
    memcpy(void* dst, const void* src, size_t size_in_bytes, MemCopyType_t type, bool sync = false);

    void memset(void* dst, int value, size_t size_in_bytes);

    void setThreadPerBlock(size_t thread_per_block)
    {
        m_thread_per_block = thread_per_block;
    }

    size_t threadsNum()
    {
        return m_thread_per_block;
    }

    size_t blocksNum(const size_t total_threads);

    int majorSM() const
    {
        return m_properties.major;
    }

    int computeCapability()
    {
        return m_properties.major * 10 + m_properties.minor;
    }

    const cudaDeviceProp* prop() const
    {
        return &m_properties;
    }

    cudaStream_t getCurrentCUDAStream()
    {
        return m_stream;
    }

private:
    void* alloc(size_t size_in_bytes);
    void  free(void* ptr);

private:
    cudaDeviceProp m_properties;
    cudaStream_t   m_stream;
    int            m_device_id;
    size_t         m_thread_per_block = 128;
};
