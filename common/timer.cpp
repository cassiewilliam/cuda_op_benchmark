#include <numeric>
#include <algorithm>
#include <math.h>

#include "log.hpp"
#include "timer.hpp"

std::unordered_map<size_t, cudaEvent_t>        Timer::g_anchors = {};
std::unordered_map<size_t, std::vector<float>> Timer::g_times = {};
int                                            Timer::g_interval = 1;

void Timer::tic(const std::string &name)
{
    size_t key = std::hash<std::string>{}(name);
    cudaEvent_t start;
    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventRecord(start));
    g_anchors[key] = start;
}

void Timer::toc(const std::string &name)
{
    size_t key = std::hash<std::string>{}(name);

    cudaEvent_t stop;
    cudaCheck(cudaEventCreate(&stop));
    cudaCheck(cudaEventRecord(stop));
    cudaCheck(cudaEventSynchronize(stop));

    float milliseconds = 0;
    cudaCheck(cudaEventElapsedTime(&milliseconds, g_anchors[key], stop));

    g_times[key].push_back(milliseconds);

    if (g_times[key].size() % g_interval == 0)
    {
        auto sum = std::accumulate(g_times[key].begin(), g_times[key].end(), 0.0);
        auto mean = sum / g_interval;

        double sum_square = 0.0;

        std::for_each(g_times[key].begin(), g_times[key].end(), [&](const float time_obj)
                      { sum_square += (time_obj - mean) * (time_obj - mean); });
        double stdev = (g_interval > 1) ? std::sqrt(sum_square / (g_interval - 1)) : 0.0f;

        g_times[key].clear();

        LOGI("%16s, TimerStatstic: average: %.3f(ms), stddev: %.3f", name.c_str(), mean, stdev);
    }
}