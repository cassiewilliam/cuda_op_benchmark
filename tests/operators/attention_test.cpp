#include "creator_factory.hpp"
#include "operator.hpp"
#include "test_suite.hpp"

static void reference_attention(Operator::TensorVector inputs, Operator::TensorVector outputs)
{
    
}

class AttentionTest : public TestCase
{
public:
    virtual ~AttentionTest() = default;
    virtual bool run(int precision)
    {
        auto map     = CreatorFactory::gCreator();
        auto op_iter = map->find(OpType::OpType_Attention);

        if (op_iter == map->end())
        {
            printf("Attention Op not register\n");
            return false;
        }

        constexpr int run_times = 1;

        auto flash_attention = op_iter->second->onCreate(KernelType::KT_FlashAttention);

        int batch_size     = 1;
        int seq_length     = 1024;
        int num_heads      = 16;
        int head_dimension = 256;

        Operator::TensorVector inputs;
        Operator::TensorVector outputs;

        auto shape = std::vector<int>{batch_size, seq_length, num_heads, head_dimension};

        auto q_tensor = Tensor::create(shape, MemoryType::GPU, DataType::FLOAT16);
        q_tensor->fillData();
        inputs.emplace_back(q_tensor);

        auto k_tensor = Tensor::create(shape, MemoryType::GPU, DataType::FLOAT16);
        k_tensor->fillData();
        inputs.emplace_back(k_tensor);

        auto v_tensor = Tensor::create(shape, MemoryType::GPU, DataType::FLOAT16);
        v_tensor->fillData();
        inputs.emplace_back(v_tensor);

        auto o_tensor = Tensor::create(shape, MemoryType::GPU, DataType::FLOAT16);
        outputs.emplace_back(o_tensor);

        flash_attention->onResize(inputs, outputs);

        runFuncWithTimeProfiler("attention", run_times, false, [&] {
            flash_attention->onExecute(inputs, outputs);
        });

        flash_attention->onFinish(inputs, outputs);

        return true;
    }
};

TestSuiteRegister(AttentionTest, "op/attention");