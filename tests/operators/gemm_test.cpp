#include "test_suite.hpp"
#include "creator_factory.hpp"
#include "operator.hpp"

static void reference_hgemm(bool          A_transpose,
                            bool          B_transpose,
                            int           m,
                            int           n,
                            int           k,
                            float         alpha,
                            const Tensor &A,
                            const Tensor &B,
                            float         beta,
                            const Tensor &C,
                            Tensor &      D)
{
    half * a_ptr = static_cast<half *>(A.data());
    half * b_ptr = static_cast<half *>(B.data());
    float *c_ptr = static_cast<float *>(C.data());
    float *d_ptr = static_cast<float *>(D.data());

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0;
            for (int t = 0; t < k; ++t)
            {
                half half_a = a_ptr[i * k + t];
                half half_b = b_ptr[j * k + t];
                sum += float(half_a) * float(half_b);
            }
            d_ptr[i * n + j] = alpha * sum + beta * c_ptr[i * n + j];
        }
    }
}

class GemmTest : public TestCase
{
public:
    virtual ~GemmTest() = default;
    virtual bool run(int flag)
    {
        auto map = CreatorFactory::gCreator();
        auto op_iter = map->find(OpType::OpType_Gemm);

        if (op_iter == map->end())
        {
            printf("Gemm Op not register\n");
            return false;
        }

        auto &factory = op_iter->second;

        auto naive_hgemm_op = factory->onCreate(KernelType::KT_HGEMM_Naive);
        auto wmma_hgemm_op = factory->onCreate(KernelType::KT_HGEMM_WMMA);
        auto mmaptx_hgemm_op = factory->onCreate(KernelType::KT_HGEMM_MMA_PTX);

        constexpr int run_times = 30;

        // NOTE: m, n, k 都能被16整除
        // int m = 128;
        // int n = 64;
        // int k = 128;

        int m = 4096;
        int n = 4096;
        int k = 4096;

        float alpha = 2.0f;
        float beta = 0.0f;

        Operator::TensorVector inputs;
        Operator::TensorVector outputs_naive, outputs_wmma, outputs_mmaptx;

        auto A = std::make_shared<Tensor>(std::vector<int>{1, 1, m, k}, MemoryType::CPU, DataType::FLOAT16);
        auto B = std::make_shared<Tensor>(std::vector<int>{1, 1, n, k}, MemoryType::CPU, DataType::FLOAT16);
        auto C = std::make_shared<Tensor>(std::vector<int>{1, 1, m, n}, MemoryType::CPU, DataType::FLOAT32);
        auto P = std::make_shared<Tensor>(std::vector<int>{1, 1, 1, 2}, MemoryType::CPU, DataType::FLOAT32);

        // NOTE: 输入参数
        ((float *)P->data())[0] = alpha;
        ((float *)P->data())[1] = beta;

        A->fillData();
        B->fillData();
        C->fillData();
        inputs.emplace_back(A);
        inputs.emplace_back(B);
        inputs.emplace_back(C);
        inputs.emplace_back(P);

        auto ref_D = std::make_shared<Tensor>(std::vector<int>{1, 1, m, n}, MemoryType::CPU, DataType::FLOAT32);
        if (flag & 0x1)
        {
            reference_hgemm(false, false, m, n, k, alpha, *A, *B, beta, *C, *ref_D);
        }

        auto naive_D = std::make_shared<Tensor>(std::vector<int>{1, 1, m, n}, MemoryType::GPU, DataType::FLOAT32);
        outputs_naive.push_back(naive_D);

        auto wmma_D = std::make_shared<Tensor>(std::vector<int>{1, 1, m, n}, MemoryType::GPU, DataType::FLOAT32);
        outputs_wmma.push_back(wmma_D);

        auto mmaptx_D = std::make_shared<Tensor>(std::vector<int>{1, 1, m, n}, MemoryType::GPU, DataType::FLOAT32);
        outputs_mmaptx.push_back(mmaptx_D);

        A->toCUDA();
        B->toCUDA();
        C->toCUDA();

        runFuncWithTimeProfiler("naive_hgemm", run_times, false, [&] {
            naive_hgemm_op->onExecute(inputs, outputs_naive);
        });

        runFuncWithTimeProfiler("wmma_hgemm", run_times, false, [&] {
            wmma_hgemm_op->onExecute(inputs, outputs_wmma);
        });

        runFuncWithTimeProfiler("mmaptx_hgemm", run_times, false, [&] {
            mmaptx_hgemm_op->onExecute(inputs, outputs_mmaptx);
        });

        if (flag & 0x1)
        {
            naive_D->toCPU();
            CE_CHECK(compareTensor(*ref_D, *naive_D, "naive_hgemm"), "compare tensor error");

            wmma_D->toCPU();
            CE_CHECK(compareTensor(*ref_D, *wmma_D, "wmma_hgemm"), "compare tensor error");

            mmaptx_D->toCPU();
            CE_CHECK(compareTensor(*ref_D, *mmaptx_D, "mmaptx_hgemm"), "compare tensor error");
        }

        return true;
    }
};

TestSuiteRegister(GemmTest, "op/gemm");