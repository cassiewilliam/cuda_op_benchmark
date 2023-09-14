#ifndef LOG_H
#define LOG_H

#include <string.h>

#define __FILENAME__ ""

#ifdef LOG_LEVEL
#    undef LOG_LEVEL
#endif
#define LOG_LEVEL 1

#ifdef LOG_TAG
#    undef LOG_TAG
#endif
#define LOG_TAG "CUDA"

#if defined(ANDROID)

#    include <android/log.h>
#    include <string.h>

#    ifdef LOGI
#        undef LOGI
#    endif
#    ifdef LOGD
#        undef LOGD
#    endif
#    ifdef LOGW
#        undef LOGW
#    endif
#    ifdef LOGV_IF
#        undef LOGV_IF
#    endif
#    ifdef LOGE
#        undef LOGE
#    endif

#    if LOG_LEVEL > 3
#        define LOGD(format, args...)                            \
            __android_log_print(ANDROID_LOG_DEBUG,               \
                                LOG_TAG,                         \
                                "[D][%.20s(%03d)]:" format "\n", \
                                __FILENAME__,                    \
                                __LINE__,                        \
                                ##args);
#    else
#        define LOGD(format, args...)
#    endif

#    if LOG_LEVEL > 2
#        define LOGI(format, args...)                            \
            __android_log_print(ANDROID_LOG_INFO,                \
                                LOG_TAG,                         \
                                "[I][%.20s(%03d)]:" format "\n", \
                                __FILENAME__,                    \
                                __LINE__,                        \
                                ##args);
#    else
#        define LOGI(format, args...)
#    endif

#    if LOG_LEVEL > 1
#        define LOGW(format, args...)                            \
            __android_log_print(ANDROID_LOG_WARN,                \
                                LOG_TAG,                         \
                                "[W][%.20s(%03d)]:" format "\n", \
                                __FILENAME__,                    \
                                __LINE__,                        \
                                ##args);
#    else
#        define LOGW(format, args...)
#    endif

#    if LOG_LEVEL > 0
#        define LOGE(format, args...)                            \
            __android_log_print(ANDROID_LOG_ERROR,               \
                                LOG_TAG,                         \
                                "[E][%.20s(%03d)]:" format "\n", \
                                __FILENAME__,                    \
                                __LINE__,                        \
                                ##args);
#    else
#        define LOGE(format, args...)
#    endif

#    define LOGV_IF(enable, format, args...)                         \
        do                                                           \
        {                                                            \
            if (enable)                                              \
            {                                                        \
                __android_log_print(ANDROID_LOG_INFO,                \
                                    LOG_TAG,                         \
                                    "[V][%.20s(%03d)]:" format "\n", \
                                    __FILENAME__,                    \
                                    __LINE__,                        \
                                    ##args);                         \
            }                                                        \
        } while (0)

#else

#    include <stdio.h>

#    define LOGI(format, ...) fprintf(stdout, "CUDA[I]" format "\n", ##__VA_ARGS__)
#    define LOGW(format, ...) \
        fprintf(stderr, "CUDA[W]: %s line %d: " format "\n", __FILENAME__, __LINE__, ##__VA_ARGS__)
#    define LOGE(format, ...) \
        fprintf(stderr, "CUDA[E]: %s line %d: " format "\n", __FILENAME__, __LINE__, ##__VA_ARGS__)
#    define LOGD(format, ...) \
        fprintf(stderr, "CUDA[D]: %s line %d: " format "\n", __FILENAME__, __LINE__, ##__VA_ARGS__)

#endif

#endif /* LOG_H_ */
