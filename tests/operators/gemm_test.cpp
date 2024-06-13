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
    virtual bool run(std::vector<int> flags)
    {
        int flag = (flags.size() > 0) ? flags[0] : 0;
        bool perf_mode = (flag & 0x1);
        auto map = CreatorFactory::gCreator();
        auto op_iter = map->find(OpType::OpType_Gemm);

        if (op_iter == map->end())
        {
            printf("Gemm Op not register\n");
            return false;
        }
  
        auto &factory = op_iter->second;

        auto bert_hgemm_op = factory->onCreate(KernelType::KT_HGEMM_BERT);
        auto epilogue_hgemm_op = factory->onCreate(KernelType::KT_HGEMM_ADDBIAS);

        const int run_times = perf_mode ? 100 : 1;

        int m = 128 * 8;
        int k = 1024;
        int n = 1024 * 3;

        float alpha = 1.0f;
        float beta = 0.0f;

        Operator::TensorVector inputs;
        Operator::TensorVector outputs_bert, outputs_epilogue;

        auto A = std::make_shared<Tensor>(std::vector<int>{1, 1, m, k}, MemoryType::CPU, DataType::FLOAT16);
        auto B = std::make_shared<Tensor>(std::vector<int>{1, 1, k, n}, MemoryType::CPU, DataType::FLOAT16);
        auto C = std::make_shared<Tensor>(std::vector<int>{1, 1, m, n}, MemoryType::CPU, DataType::FLOAT32);
        auto Bias = std::make_shared<Tensor>(std::vector<int>{1, 1, 1, n}, MemoryType::CPU, DataType::FLOAT32);
        auto P = std::make_shared<Tensor>(std::vector<int>{1, 1, 1, 2}, MemoryType::CPU, DataType::FLOAT32);

        // NOTE: 输入参数
        ((float *)P->data())[0] = alpha;
        ((float *)P->data())[1] = beta;

        A->fillData();
        B->fillData();
        C->fillData();
        Bias->fillData();
        inputs.emplace_back(A);
        inputs.emplace_back(B);
        inputs.emplace_back(C);
        inputs.emplace_back(Bias);
        inputs.emplace_back(P);

        auto ref_D = std::make_shared<Tensor>(std::vector<int>{1, 1, m, n}, MemoryType::CPU, DataType::FLOAT16);
        if (!perf_mode)
        {
            reference_hgemm(false, false, m, n, k, alpha, *A, *B, beta, *C, *ref_D);
        }

        auto bert_D = std::make_shared<Tensor>(std::vector<int>{1, 1, m, n}, MemoryType::GPU, DataType::FLOAT16);
        outputs_bert.push_back(bert_D);

        auto epilogue_D = std::make_shared<Tensor>(std::vector<int>{1, 1, m, n}, MemoryType::GPU, DataType::FLOAT16);
        outputs_epilogue.push_back(epilogue_D);

        A->toCUDA();
        B->toCUDA();
        C->toCUDA();
        Bias->toCUDA();

       bert_hgemm_op->onResize(inputs, outputs_bert);
       epilogue_hgemm_op->onResize(inputs, outputs_epilogue);

        runFuncWithTimeProfiler("bert_hgemm", run_times, perf_mode, [&] {
            bert_hgemm_op->onExecute(inputs, outputs_bert);
        });

        runFuncWithTimeProfiler("epilogue_hgemm", run_times, perf_mode, [&] {
            epilogue_hgemm_op->onExecute(inputs, outputs_epilogue);
        });

        if (!perf_mode)
        {
            bert_D->toCPU();
            CE_CHECK(compareTensor(*ref_D, *bert_D, "bert_hgemm"), "compare tensor error");

            epilogue_D->toCPU();
            CE_CHECK(compareTensor(*ref_D, *epilogue_D, "epilogue_hgemm"), "compare tensor error");
        }
        //*/
        return true;
    }
};

TestSuiteRegister(GemmTest, "op/gemm");