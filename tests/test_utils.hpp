#pragma once

#include "timer.hpp"
#include "tensor.hpp"

#define EPSILON 1e-1

#include <functional>

template<typename T>
static bool isEqual(std::vector<T> const &v1, std::vector<T> const &v2)
{
    return (v1.size() == v2.size() &&
            std::equal(v1.begin(), v1.end(), v2.begin()));
}

template<typename T>
static bool isEqual(const T &lhs, const T &rhs)
{
    return (lhs == rhs);
}

template <>
bool isEqual<float>(const float & lhs, const float & rhs)
{
    return (fabs(lhs - rhs) <= EPSILON);
}

template <>
bool isEqual<half>(const half& lhs, const half & rhs)
{
    return (fabs((float)lhs - (float)rhs) <= EPSILON);
}

static void runFuncWithTimeProfiler(std::string           name,
                                    int                   run_times,
                                    bool                  warp_up,
                                    std::function<void()> func)
{
    if (warp_up)
    {
        // NOTE: warm up run before profile
        for (int i = 0; i < 5; ++i)
        {
            func();
        }
    }

    Timer::g_interval = run_times;

    // NOTE: run profile
    for (int i = 0; i < run_times; ++i)
    {
        Timer::tic(name);
        func();
        Timer::toc(name);
    }
}

static bool compareTensor(Tensor& lhs, Tensor& rhs, const char* msg = "")
{
    if (lhs.dtype() != rhs.dtype())
    {
        LOGE("dtype not equal: %d, %d", lhs.dtype(), rhs.dtype());
        return false;
    }

    if (lhs.mtype() != rhs.mtype())
    {
        LOGE("mtype not equal: %d, %d", lhs.mtype(), rhs.mtype());
        return false;
    }

    if (lhs.format() != rhs.format())
    {
        LOGE("format not equal: %d, %d", lhs.format(), rhs.format());
        return false;
    }

    if (!lhs.cpu_data() || !rhs.cpu_data())
    {
        LOGE("only cpu data can compare, please call toCPU() first\n");
        return false;
    }

    auto gt_shapes = lhs.shapes();
    auto tt_shapes = rhs.shapes();

    if (!isEqual(gt_shapes, tt_shapes))
    {
        LOGE("tensor shapes not equal\n");
        return false;
    }

    auto gt_strides = lhs.strides();
    auto tt_strides = lhs.strides();

    if (!isEqual(gt_strides, tt_strides))
    {
        LOGE("tensor strides not equal\n");
        return false;
    }

    for (int i = 0; i < gt_shapes[0]; ++i)
    {
        for (int j = 0; j < gt_shapes[1]; ++j)
        {
            for (int k = 0; k < gt_shapes[2]; ++k)
            {
                for (int m = 0; m < gt_shapes[3]; ++m)
                {
                    auto offset = i * gt_strides[0] + j * gt_strides[1] + k * gt_strides[2] + m * gt_strides[3];
                    if (lhs.dtype() == DataType::FLOAT16)
                    {
                        auto lhs_elem = ((half *)lhs.data())[offset];
                        auto rhs_elem = ((half *)rhs.data())[offset];

                        if (!isEqual<half>(lhs_elem, rhs_elem))
                        {
                            LOGE("pos: %d, %d, %d, %d elem not equal, %.2f, %.2f\n",
                                i, j, k, m, 
                                float(lhs_elem),
                                float(rhs_elem));
                            return false;
                        }
                    }
                    else if (lhs.dtype() == DataType::FLOAT32)
                    {
                        auto lhs_elem = ((float *)lhs.data())[offset];
                        auto rhs_elem = ((float *)rhs.data())[offset];

                        if (!isEqual<float>(lhs_elem, rhs_elem))
                        {
                            LOGE("pos: %d, %d, %d, %d elem not equal, %.2f, %.2f\n", i, j, k, m, lhs_elem, rhs_elem);
                            return false;
                        }
                    }
                    else if (lhs.dtype() == DataType::INT32)
                    {
                        auto lhs_elem = ((int *)lhs.data())[offset];
                        auto rhs_elem = ((int *)rhs.data())[offset];

                        if (!isEqual<int>(lhs_elem, rhs_elem))
                        {
                            LOGE("pos: %d, %d, %d, %d elem not equal, %d, %d\n", i, j, k, m, lhs_elem, rhs_elem);
                            return false;
                        }
                    }
                    else
                    {
                        LOGE("not supported!");
                    }
                }
            }
        }
    }

    LOGI("%16s, Correctness: Tenosr compare success", msg);
    return true;
}
