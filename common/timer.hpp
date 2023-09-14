#pragma once

#include <stdlib.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <string>

#include "cuda_runtime.hpp"

class Timer
{
public:
    static void tic(const std::string &name);
    static void toc(const std::string &name);

public:
    static std::unordered_map<size_t, cudaEvent_t>        g_anchors;
    static std::unordered_map<size_t, std::vector<float>> g_times;

    static int g_interval;
};
