#pragma once

#include "log.hpp"

#include <cassert>
#include <memory>


// Math
#ifndef UP_DIV
#define UP_DIV(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y))
#endif
#ifndef ROUND_UP
#define ROUND_UP(x, y) (((int)(x) + (int)(y) - (1)) / (int)(y) * (int)(y))
#endif
#ifndef ALIGN_UP4
#define ALIGN_UP4(x) ROUND_UP((x), 4)
#endif

#ifndef ALIGN_UP8
#define ALIGN_UP8(x) ROUND_UP((x), 8)
#endif
#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif
#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif
#ifndef ABS
#define ABS(x) ((x) > (0) ? (x) : (-(x)))
#endif

#define CE_CHECK(success, log)                             \
    if (!(success))                                        \
    {                                                      \
        LOGE("Check failed: %s ==> %s\n", #success, #log); \
    }

inline std::shared_ptr<uint8_t> AllocBufferWithByteSize(int data_byte_size)
{
    return std::shared_ptr<uint8_t>(new uint8_t[data_byte_size], std::default_delete<uint8_t[]>());
};