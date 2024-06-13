cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

project(CUDABench)
message(STATUS "CMake Project Name : ${PROJECT_NAME}")

option(CUDA_PROFILE "Enable CUDA profile" OFF)
option(CUTLASS_ENABLE "Enable CUTLASS include" OFF)

if(NOT CMAKE_SYSTEM_NAME MATCHES "^Linux")
    message(FATAL_ERROR "Currently Only Support Linux Platform")
endif()

# NOTE(Yang Min): for half operator, architectures must greater than 70
# 89 means build with RTX4090 GPU
# 80 means build with A100 GPU
# 70 means build with V100 GPU
# 75 means build with T4 GPU
# NOTE: 70 需要替换cutlass版本到3.1.0
set(CMAKE_CUDA_ARCHITECTURES "80")

find_package(CUDA REQUIRED)
enable_language(CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -ldl") # -Xptxas -v

set(EXTRA_LIBS "")

if(CUDA_PROFILE)
    set(EXTRA_LIBS -lnvToolsExt)
endif()

if(CUDA_FOUND)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -D_FORCE_INLINES -Wno-deprecated-gpu-targets -w ${EXTRA_LIBS}")

    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0")
    else()
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
    endif()

    if(WIN32)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler /FS")
    endif()

    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SelectCudaComputeArch.cmake)
    CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS ${CUDA_ARCHS})

    list(LENGTH CUDA_ARCH_FLAGS_readable_code arch_count)

    # Current Supported Arch List
    if(${arch_count} EQUAL 1)
        set(support_archs 60 61 62 70 72 75 80 86 89 90)
        list(FIND support_archs ${CUDA_ARCH_FLAGS_readable_code} list_index)

        if(${list_index} EQUAL -1)
            message(FATAL_ERROR "Please add your own sm arch ${CUDA_ARCH_FLAGS_readable_code} to CmakeLists.txt!")
        endif()
    endif()

    if((CUDA_VERSION VERSION_GREATER "12.0") OR(CUDA_VERSION VERSION_EQUAL "12.2"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_89,code=sm_89")
    elseif((CUDA_VERSION VERSION_GREATER "11.2") OR(CUDA_VERSION VERSION_EQUAL "11.2"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
    elseif((CUDA_VERSION VERSION_GREATER "10.2") OR(CUDA_VERSION VERSION_EQUAL "10.2"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
    elseif((CUDA_VERSION VERSION_GREATER "10.1") OR(CUDA_VERSION VERSION_EQUAL "10.1"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_72,code=sm_72")
    elseif((CUDA_VERSION VERSION_GREATER "8.0") OR(CUDA_VERSION VERSION_EQUAL "8.0"))
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
    endif()

    # Limit minimum cuda version for each archs
    if(${arch_count} EQUAL 1)
        if((CUDA_ARCH_FLAGS_readable_code VERSION_GREATER "80") OR(CUDA_ARCH_FLAGS_readable_code VERSION_EQUAL "80"))
            if(CUDA_VERSION VERSION_LESS "11.2")
                message(FATAL_ERROR "Please update cuda version to 11.2 or higher!")
            endif()
        endif()

        if((CUDA_ARCH_FLAGS_readable_code VERSION_GREATER "75") OR(CUDA_ARCH_FLAGS_readable_code VERSION_EQUAL "75"))
            if(CUDA_VERSION VERSION_LESS "10.2")
                message(FATAL_ERROR "Please update cuda version to 10.2 or higher!")
            endif()
        endif()

        if((CUDA_ARCH_FLAGS_readable_code VERSION_GREATER "70") OR(CUDA_ARCH_FLAGS_readable_code VERSION_EQUAL "70"))
            if(CUDA_VERSION VERSION_LESS "10.1")
                message(FATAL_ERROR "Please update cuda version to 10.1 or higher!")
            endif()
        endif()
    endif()

    message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
        " archs: ${CUDA_ARCH_FLAGS_readable})")
else()
    message(FATAL_ERROR "CUDA not found >= ${CUDA_MIN_VERSION} required)")
endif()

file(GLOB_RECURSE CUDA_SRC ${PROJECT_SOURCE_DIR}/source/*
    ${PROJECT_SOURCE_DIR}/common/*)

message(STATUS "info ${CUDA_SRC}")
message(STATUS "message ${CUDA_NVCC_FLAGS} !!!!!!!!!!!")

# cuda_add_library(CUDA_LIBS SHARED ${CUDA_SRC})

# NOTE: 包含所有需要测试的文件夹头文件
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/source
    ${PROJECT_SOURCE_DIR}/source/attention
    ${PROJECT_SOURCE_DIR}/tests
    ${PROJECT_SOURCE_DIR}/common
    ${PROJECT_SOURCE_DIR}/3rdparty
    ${PROJECT_SOURCE_DIR}/utils
    ${CUDA_INCLUDE_DIRS}
)

if(CUTLASS_ENABLE)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass/include)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/cutlass)
endif(CUTLASS_ENABLE)

file(GLOB_RECURSE TEST_SRC ${PROJECT_SOURCE_DIR}/tests/*.cpp)

add_executable(${PROJECT_NAME} ${TEST_SRC} ${CUDA_SRC} ${CUTLASS})

# NOTE: add pthread for std::call_once
# https://stackoverflow.com/questions/65335620/terminate-called-after-throwing-an-instance-of-stdsystem-error
# target_link_libraries(${PROJECT_NAME} ${CUDA_LIBS} -pthread dl)
target_link_libraries(${PROJECT_NAME} -pthread -lcudart -lcublas -lcublasLt -lcurand -fopenmp)

if(CUDA_PROFILE)
    target_compile_options(${PROJECT_NAME} PRIVATE -DCUDA_PROFILE)
    target_link_libraries(${PROJECT_NAME} ${CUDA_INCLUDE_DIRS}/../lib/libnvToolsExt.so)
endif()
