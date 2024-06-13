# cuda_op_benchmark

在实际的推理引擎开发中，更多的是深度学习算子实现，所以本仓库目标是以实现高效深度学习算子为切入点，对CUDA编程，以及CUDA优化和概念进行一个介绍。
本仓库旨在通过算子Benchmark来对CUDA编程进行学习，框架采用CMake进行编译，也可以比较方便的进行代码组织，可以使得小伙伴们专注在算子的优化上面。

## 个人开发服务器配置环境说明
- RTX4090

## 算子完成情况

|  Op Name  |   Naive   | Version2          | Version 3  | Version 4      | Version 5   | Status      |
| ------    | --------- | ----------------- | ---------  | ---------      | ---------   | -------     |
|  GEMM     |     ✅    | Wmma ✅           | mma-ptx ✅ | mma-ptx-opt ❌ |  cutlass ❌ |  Doing      |
| Attention |           | FlashAttention ❌ |            |                 |             |    Doing   |

## 使用方法

### 1. 编译
在当前工程目录运行
#### 编译
```
bash scripts/build_linux.sh
```

#### 运行
```
# 运行gemm op的测试，只测试cuda kernel的耗时
./build/CUDABench op/gemm

# 运行gemm op的测试，测试cuda kernel的耗时，并且测试其正确性
./build/CUDABench op/gemm 1

# 运行gemm op的测试，测试cuda kernel的耗时，并且测试其正确性，并且将所有log输出到log.txt
./build/CUDABench op/gemm 1 > log.tx 2>&1
```

## Reference
- [1] https://github.com/Bruce-Lee-LY/cuda_hgemm
- [2] https://github.com/alibaba/MNN
- [3] https://github.com/NVIDIA/cutlass
- [4] https://github.com/pybind/pybind11