#pragma once

#include "timer.hpp"
#include "tensor.hpp"

#define EPSILON 1e-5

#include <functional>

static int randomgen(int min, int max) //Pass in range
{
    // srand(time(NULL));  //Changed from rand(). srand() seeds rand for you.
    int random = rand() % (max - min) + min;
    return random;
}

template<typename T1, typename T2>
void compareTwoTensor(const T1* pred, const T2* ref, const int size, const int print_size = 0, const std::string filename = "")
{
    FILE* fd = nullptr;
    if (filename != "") {
        fd = fopen(filename.c_str(), "w");
        fprintf(fd, "| %10s | %10s | %10s | %10s | \n", "pred", "ref", "abs_diff", "rel_diff(%%)");
    }

    if (print_size > 0) {
        LOGI("  id |   pred  |   ref   |abs diff | rel diff (%%) |");
    }
    float max_abs_diff  = 0.0f;
    float mean_abs_diff = 0.0f;
    float mean_rel_diff = 0.0f;
    int   count         = 0;

    #pragma omp parallel for
    for (int i = 0; i < size; i++) {
        if (i < print_size) {
            LOGI("%4d | % 6.4f | % 6.4f | % 6.4f | % 7.4f |",
                        i,
                        (float)pred[i],
                        (float)ref[i],
                        abs((float)pred[i] - (float)ref[i]),
                        abs((float)pred[i] - (float)ref[i]) / (abs((float)ref[i]) + 1e-6f) * 100.f);
        }
        if ((float)pred[i] == 0) {
            continue;
        }
        count += 1;
        mean_abs_diff += abs((float)pred[i] - (float)ref[i]);
        mean_rel_diff += abs((float)pred[i] - (float)ref[i]) / (abs((float)ref[i]) + 1e-6f) * 100.f;

        max_abs_diff = std::max(max_abs_diff, (float)pred[i] - (float)ref[i]);

        if (fd != nullptr) {
            fprintf(fd,
                    "| %10.5f | %10.5f | %10.5f | %11.5f |\n",
                    (float)pred[i],
                    (float)ref[i],
                    abs((float)pred[i] - (float)ref[i]),
                    abs((float)pred[i] - (float)ref[i]) / (abs((float)ref[i]) + 1e-6f) * 100.f);
        }
    }
    mean_abs_diff = (count == 0) ? 0.0f : mean_abs_diff / (float)count;
    mean_rel_diff = (count == 0) ? 0.0f :mean_rel_diff / (float)count;
    LOGI("%16s, TensorCompare: size: %d, count: %d, mean_abs_diff: % 6.4f, mean_rel_diff: % 6.4f (%%), max_abs_diff: % 6.4f", 
         filename.substr(0, filename.find('.')).c_str(), size, count, mean_abs_diff, mean_rel_diff, max_abs_diff);

    if (fd != nullptr) {
        fprintf(fd, "size: %d, count: %d, mean_abs_diff: % 6.4f, mean_rel_diff: % 6.4f (%%), max_abs_diff: % 6.4f", 
                size, count, mean_abs_diff, mean_rel_diff, max_abs_diff);
        fclose(fd);
    }
}

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

    if (lhs.dtype() == DataType::FLOAT16)
    {
        compareTwoTensor<half, half>(((half *)rhs.data()), ((half *)lhs.data()), lhs.size(), 0, std::string(msg) + ".txt");
    }
    else if (lhs.dtype() == DataType::FLOAT32)
    {
        compareTwoTensor<float, float>(((float *)rhs.data()), ((float *)lhs.data()), lhs.size(), 0, std::string(msg) + ".txt");
    }
    else if (lhs.dtype() == DataType::INT32)
    {
        compareTwoTensor<int, int>(((int *)rhs.data()), ((int *)lhs.data()), lhs.size(), 0, std::string(msg) + ".txt");
    }
    else
    {
        LOGE("not supported!");
    }

    return true;
}
