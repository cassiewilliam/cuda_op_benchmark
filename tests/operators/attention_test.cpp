#include "creator_factory.hpp"
#include "operator.hpp"
#include "test_suite.hpp"

typedef std::shared_ptr<Tensor> TensorPtr;

// with padding
static void reference_attention(const TensorPtr& QKV, const TensorPtr& M, const TensorPtr SEQ_LEN, TensorPtr& O, int b, int s, int h, int d)
{
    // NOTE: o = softmax((q * k^T) / sqrt(head_dim) + mask) * v
    //       1. q, k, v重排
    //       2. 计算qk = q * k^T
    //       3. 计算w = softmax(qk / sqrt(head_dim) + mask)
    //       4. 计算o_transpose = w * v
    //       5. o = 重排(o_transpose)
    printf("%d, %d, %d, %d\n", b, s, h, d);
    auto qkv_shape = std::vector<int>{b, s, h, d};
    auto q_buf = Tensor::create(qkv_shape, MemoryType::CPU, DataType::FLOAT16);
    auto k_buf = Tensor::create(qkv_shape, MemoryType::CPU, DataType::FLOAT16);
    auto v_buf = Tensor::create(qkv_shape, MemoryType::CPU, DataType::FLOAT16);

    memset(q_buf->data(), 0, b * h * s * d * sizeof(half));
    memset(k_buf->data(), 0, b * h * s * d * sizeof(half));
    memset(v_buf->data(), 0, b * h * s * d * sizeof(half));
    // Step1: q, k, v重排
    // 输入shape为[batch, seq_len, num_heads, head_dim]
    // 目标shape为[batch, num_heads, seq_len, head_dim]
    int offset = 0;
    for (int i = 0; i < b; ++i)
    {
        int cu_seq_len = ((int *)(SEQ_LEN->data()))[i + 1] - ((int *)(SEQ_LEN->data()))[i];
        for (int j = 0; j < cu_seq_len; ++j)
        {
            for(int k = 0; k < h; ++k)
            {
                for (int l = 0; l < d; ++l)
                {
                    int batch_id = i;
                    int seq_id = j;
                    int head_id = k;
                    int id_in_head = l;

                    int target_idx = batch_id * (h * s * d) + head_id * (s * d) + seq_id * (d) + id_in_head;
                    int source_idx = offset * (h * d * 3)   + head_id * (d) + id_in_head;

                    ((half *)(q_buf->data()))[target_idx] = ((half *)(QKV->data()))[source_idx];
                    ((half *)(k_buf->data()))[target_idx] = ((half *)(QKV->data()))[h * d + source_idx];
                    ((half *)(v_buf->data()))[target_idx] = ((half *)(QKV->data()))[h * d * 2 + source_idx];
                }
            }
            offset++;
        }
    }

    if (0)
    {
        for (int i = 0; i < 2000; ++i)
        {
            printf("cpu1: %d, %.2f, %.2f\n", i, (float)(((half *)q_buf->data())[i]), (float)(((half *)k_buf->data())[i]));
        }
    }

    // Step2: 计算qk = qk^T
    // qk的shape为[batch, num_heads, seq_len, head_dim]
    // k^T的shape为[batch, num_heads, head_dim, seq_len]，可以不用真实转置，直接按列主序取对应位置就好
    auto qk_buf = Tensor::create(std::vector<int>{b, h, s, s}, MemoryType::CPU);
    for (int i = 0; i < b; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            int qk_offset = i * (h * s * s) + j * (s * s);
            auto qk_ptr = (half *)(qk_buf->data()) + qk_offset;
            for (int k = 0; k < s; ++k)
            {
                int q_offset = i * (h * s * d) + j * (s * d) + k * d;
                auto q_ptr = (half *)(q_buf->data()) + q_offset;
                for (int l = 0; l < s; ++l)
                {
                    int k_offset = i * (h * s * d) + j * (s * d) + l * d;
                    auto k_ptr = (half *)(k_buf->data()) + k_offset;
                    float acc = 0.0f;
                    for (int m = 0; m < d; ++m)
                    {
                        acc += (float)(q_ptr[m]) * (float)(k_ptr[m]);
                    }
                    qk_ptr[k * s + l] = half(acc);
                }
            }
        }
    }


    if (0)
    {
        for (int i = 0; i < 2000; ++i)
        {
            printf("cpu1: %d, %.2f\n", i, (float)(((half *)qk_buf->data())[i]));
        }
    }

    // Step3: 计算w = softmax(qk / sqrt(head_dim) + mask)
    // qk_buf的shape为[batch, num_heads, seq_len, seq_len];
    for (int i = 0; i < b; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            for (int k = 0; k < s; ++k)
            {
                int qk_offset = i * (h * s * s) + j * (s * s) + k * s;
                auto qk_ptr = (half *)(qk_buf->data()) + qk_offset;

                float max_val = -10000.0f;

                auto mask_ptr = ((half *)M->data()) + i * (s * s) + k * s;
                
                const float scalar = 1.0f / std::sqrt(d);
                // Softmax Step1: 计算max
                for (int l = 0; l < s; ++l)
                {
                    float mask_val = (float)(mask_ptr[l]);
                    mask_val = (1.0f - mask_val) * -10000.0f;
                    float tmp = (float)(qk_ptr[l]) * scalar + mask_val;
                    max_val = std::max(max_val, tmp);
                }

                // Softmax Step2: 计算exp sum
                float sum_val = 0.0f;
                for (int l = 0; l < s; ++l)
                {
                    float mask_val = (float)(mask_ptr[l]);
                    mask_val = (1.0f - mask_val) * -10000.0f;
                    float tmp = (float)(qk_ptr[l]) * scalar + mask_val;

                    sum_val += std::exp(tmp - max_val);
                }

                // Softmax Step3: 计算prob
                for (int l = 0; l < s; ++l)
                {
                    float mask_val = (float)(mask_ptr[l]);
                    mask_val = (1.0f - mask_val) * -10000.0f;
                    float tmp = (float)(qk_ptr[l]) * scalar + mask_val;

                    qk_ptr[l] = half(std::exp(tmp - max_val) / sum_val);
                }
            }
        }
    }

    if (0)
    {
        for (int i = 0; i < 2000; ++i)
        {
            printf("cpu1: %d, %.4f\n", i, (float)(((half *)qk_buf->data())[i]));
        }
    }

    auto o_buf = Tensor::create(std::vector<int>{b, h, s, d}, MemoryType::CPU, O->dtype());
    memset(o_buf->data(), 0, b * h * s * d * sizeof(half));
    // Step4: 计算o_transpose = w * v
    // w的shape为[b, h, s, s]
    // v的shape为[b, h, s, d]
    for (int i = 0; i < b; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            int o_offset = i * (h * s * d) + j * (s * d);
            auto o_ptr = (half *)(o_buf->data()) + o_offset;
            for (int k = 0; k < s; ++k)
            {
                int qk_offset = i * (h * s * s) + j * (s * s) + k * s;
                auto qk_ptr = (half *)(qk_buf->data()) + qk_offset;

                int v_offset = i * (h * s * d) + j * (s * d);
                auto v_ptr = (half *)(v_buf->data()) + v_offset;
                for (int l = 0; l < d; ++l)
                {
                    float acc = 0.0f;
                    for (int m = 0; m < s; ++m)
                    {
                        acc += (float)(qk_ptr[m]) * (float)(v_ptr[m * d + l]);
                    }
                    o_ptr[k * d + l] = half(acc);
                }
            }
        }
    }

    // Step5: o = 重排(o_transpose)
    // 输入shape为[batch, num_heads, seq_len, head_dim]
    // 目标shape为[batch, seq_len, num_heads, head_dim]
    for (int i = 0; i < b; ++i)
    {
        for (int j = 0; j < s; ++j)
        {
            for(int k = 0; k < h; ++k)
            {
                for (int l = 0; l < d; ++l)
                {
                    int batch_id = i;
                    int seq_id = j;
                    int head_id = k;
                    int id_in_head = l;

                    int source_idx = batch_id * (h * s * d) + head_id * (s * d) + seq_id * (d) + id_in_head;
                    int target_idx = batch_id * (s * h * d) + seq_id * (h * d) + head_id * (d) + id_in_head;

                    ((half *)(O->data()))[target_idx] = ((half *)(o_buf->data()))[source_idx];
                }
            }
        }
    }
}

// with padding
static void reference_attention(const TensorPtr& Q, const TensorPtr& K, const TensorPtr& V, const TensorPtr& M, const TensorPtr& SEQ_LEN, TensorPtr& O)
{
    const int b = Q->batch();
    const int s = Q->seqlen();
    const int h = Q->numheads();
    const int d = Q->headdim();

    // NOTE: o = softmax((q * k^T) / sqrt(head_dim) + mask) * v
    //       1. q, k, v重排
    //       2. 计算qk = q * k^T
    //       3. 计算w = softmax(qk / sqrt(head_dim) + mask)
    //       4. 计算o_transpose = w * v
    //       5. o = 重排(o_transpose)
    auto qkv_shape = std::vector<int>{b, s, h, d};
    auto q_buf = Tensor::create(qkv_shape, MemoryType::CPU, Q->dtype());
    auto k_buf = Tensor::create(qkv_shape, MemoryType::CPU, K->dtype());
    auto v_buf = Tensor::create(qkv_shape, MemoryType::CPU, V->dtype());

    // Step1: q, k, v重排
    // 输入shape为[batch, seq_len, num_heads, head_dim]
    // 目标shape为[batch, num_heads, seq_len, head_dim]
    for (int i = 0; i < b; ++i)
    {
        for (int j = 0; j < s; ++j)
        {
            for(int k = 0; k < h; ++k)
            {
                for (int l = 0; l < d; ++l)
                {
                    int batch_id = i;
                    int seq_id = j;
                    int head_id = k;
                    int id_in_head = l;

                    int target_idx = batch_id * (h * s * d) + head_id * (s * d) + seq_id * (d) + id_in_head;
                    int source_idx = batch_id * (s * h * d) + seq_id * (h * d) + head_id * (d) + id_in_head;

                    ((half *)(q_buf->data()))[target_idx] = ((half *)(Q->data()))[source_idx];
                    ((half *)(k_buf->data()))[target_idx] = ((half *)(K->data()))[source_idx];
                    ((half *)(v_buf->data()))[target_idx] = ((half *)(V->data()))[source_idx];
                }
            }
        }
    }

    if (0)
    {
        for (int i = 0; i < 2000; ++i)
        {
            printf("cpu: %d, %.2f, %.2f\n", i, (float)(((half *)q_buf->data())[i]), (float)(((half *)k_buf->data())[i]));
        }
    }

    // Step2: 计算qk = qk^T
    // qk的shape为[batch, num_heads, seq_len, head_dim]
    // k^T的shape为[batch, num_heads, head_dim, seq_len]，可以不用真实转置，直接按列主序取对应位置就好
    auto qk_buf = Tensor::create(std::vector<int>{b, h, s, s}, MemoryType::CPU);
    for (int i = 0; i < b; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            int qk_offset = i * (h * s * s) + j * (s * s);
            auto qk_ptr = (half *)(qk_buf->data()) + qk_offset;
            for (int k = 0; k < s; ++k)
            {
                int q_offset = i * (h * s * d) + j * (s * d) + k * d;
                auto q_ptr = (half *)(q_buf->data()) + q_offset;
                for (int l = 0; l < s; ++l)
                {
                    int k_offset = i * (h * s * d) + j * (s * d) + l * d;
                    auto k_ptr = (half *)(k_buf->data()) + k_offset;
                    float acc = 0.0f;
                    for (int m = 0; m < d; ++m)
                    {
                        acc += (float)(q_ptr[m]) * (float)(k_ptr[m]);
                    }
                    qk_ptr[k * s + l] = half(acc);
                }
            }
        }
    }


    if (0)
    {
        for (int i = 0; i < 2000; ++i)
        {
            printf("cpu: %d, %.2f\n", i, (float)(((half *)qk_buf->data())[i]));
        }
    }

    // Step3: 计算w = softmax(qk / sqrt(head_dim) + mask)
    // qk_buf的shape为[batch, num_heads, seq_len, seq_len];
    for (int i = 0; i < b; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            for (int k = 0; k < s; ++k)
            {
                int qk_offset = i * (h * s * s) + j * (s * s) + k * s;
                auto qk_ptr = (half *)(qk_buf->data()) + qk_offset;

                float max_val = -10000.0f;

                auto mask_ptr = ((half *)M->data()) + i * (s * s) + k * s;
                
                const float scalar = 1.0f / std::sqrt(d);
                // Softmax Step1: 计算max
                for (int l = 0; l < s; ++l)
                {
                    float mask_val = (float)(mask_ptr[l]);
                    mask_val = (1.0f - mask_val) * -10000.0f;
                    float tmp = (float)(qk_ptr[l]) * scalar + mask_val;
                    max_val = std::max(max_val, tmp);
                }

                // Softmax Step2: 计算exp sum
                float sum_val = 0.0f;
                for (int l = 0; l < s; ++l)
                {
                    float mask_val = (float)(mask_ptr[l]);
                    mask_val = (1.0f - mask_val) * -10000.0f;
                    float tmp = (float)(qk_ptr[l]) * scalar + mask_val;

                    sum_val += std::exp(tmp - max_val);
                }

                // Softmax Step3: 计算prob
                for (int l = 0; l < s; ++l)
                {
                    float mask_val = (float)(mask_ptr[l]);
                    mask_val = (1.0f - mask_val) * -10000.0f;
                    float tmp = (float)(qk_ptr[l]) * scalar + mask_val;

                    qk_ptr[l] = half(std::exp(tmp - max_val) / sum_val);
                }
            }
        }
    }

    if (0)
    {
        for (int i = 0; i < 2000; ++i)
        {
            printf("cpu: %d, %.4f\n", i, (float)(((half *)qk_buf->data())[i]));
        }
    }

    auto o_buf = Tensor::create(std::vector<int>{b, h, s, d}, MemoryType::CPU, O->dtype());
    // Step4: 计算o_transpose = w * v
    // w的shape为[b, h, s, s]
    // v的shape为[b, h, s, d]
    for (int i = 0; i < b; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            int o_offset = i * (h * s * d) + j * (s * d);
            auto o_ptr = (half *)(o_buf->data()) + o_offset;
            for (int k = 0; k < s; ++k)
            {
                int qk_offset = i * (h * s * s) + j * (s * s) + k * s;
                auto qk_ptr = (half *)(qk_buf->data()) + qk_offset;

                int v_offset = i * (h * s * d) + j * (s * d);
                auto v_ptr = (half *)(v_buf->data()) + v_offset;
                for (int l = 0; l < d; ++l)
                {
                    float acc = 0.0f;
                    for (int m = 0; m < s; ++m)
                    {
                        acc += (float)(qk_ptr[m]) * (float)(v_ptr[m * d + l]);
                    }
                    o_ptr[k * d + l] = half(acc);
                }
            }
        }
    }

    // Step5: o = 重排(o_transpose)
    // 输入shape为[batch, num_heads, seq_len, head_dim]
    // 目标shape为[batch, seq_len, num_heads, head_dim]
    for (int i = 0; i < b; ++i)
    {
        for (int j = 0; j < s; ++j)
        {
            for(int k = 0; k < h; ++k)
            {
                for (int l = 0; l < d; ++l)
                {
                    int batch_id = i;
                    int seq_id = j;
                    int head_id = k;
                    int id_in_head = l;

                    int source_idx = batch_id * (h * s * d) + head_id * (s * d) + seq_id * (d) + id_in_head;
                    int target_idx = batch_id * (s * h * d) + seq_id * (h * d) + head_id * (d) + id_in_head;

                    ((half *)(O->data()))[target_idx] = ((half *)(o_buf->data()))[source_idx];
                }
            }
        }
    }
}

class AttentionTest : public TestCase
{
public:
    virtual ~AttentionTest() = default;
   virtual bool run(std::vector<int> flags)
    {
        int flag = (flags.size() > 0) ? flags[0] : 0;
        bool perf_mode = (flag & 0x1);
        
        auto map     = CreatorFactory::gCreator();
        auto op_iter = map->find(OpType::OpType_Attention);

        if (op_iter == map->end())
        {
            printf("Attention Op not register\n");
            return false;
        }

        const int run_times = perf_mode ? 100 : 1;

        auto unfused_attention = op_iter->second->onCreate(KernelType::KT_UnFusedAttention);
        auto flash_bert_attention = op_iter->second->onCreate(KernelType::KT_FlashAttention_For_T4);
        auto byte_attention = op_iter->second->onCreate(KernelType::KT_ByteTransformerAttention);

        // int batch_size     = 2;
        // int seq_length     = 2048;
        // int num_heads      = 32;
        // int head_dimension = 128;

        int batch_size     = 8;
        int min_seq_length = 80;
        int max_seq_length = 256;
        int num_heads      = 16;
        int head_dimension = 64;

        Operator::TensorVector inputs;
        Operator::TensorVector outputs_unfused, outputs_flash, outputs_byte;

        auto shape = std::vector<int>{batch_size, max_seq_length, num_heads, head_dimension};

        auto Q = Tensor::create(shape, MemoryType::CPU, DataType::FLOAT16);
        auto K = Tensor::create(shape, MemoryType::CPU, DataType::FLOAT16);
        auto V = Tensor::create(shape, MemoryType::CPU, DataType::FLOAT16);
        auto M = Tensor::create({1, batch_size, max_seq_length, max_seq_length}, MemoryType::CPU, DataType::FLOAT16);
        auto SEQ_LEN = Tensor::create({1, 1, 1, batch_size + 1}, MemoryType::CPU, DataType::INT32);

        auto Param = std::make_shared<Tensor>(std::vector<int>{1, 1, 1, 8}, MemoryType::CPU, DataType::INT32);

        // NOTE: 输入参数
        ((int *)Param->data())[0] = batch_size;
        ((int *)Param->data())[1] = max_seq_length;
        ((int *)Param->data())[2] = num_heads;
        ((int *)Param->data())[3] = head_dimension;

        bool is_random_value = flags[2] == 1;
        Q->fillData(is_random_value);
        K->fillData(is_random_value);
        V->fillData(is_random_value);

        int first_seq_len = flags[1];

        // 生成CU_SEQ_LEN的数据
        int total_seq_lens = 0;
        for (int b = 0; b < batch_size; ++b)
        {
            int cu_seq_len = randomgen(min_seq_length, max_seq_length);
            // if (b == 0)
            {
                cu_seq_len = first_seq_len;
            }
            total_seq_lens += cu_seq_len;
            ((int *)(SEQ_LEN->data()))[b + 1] = total_seq_lens;
        }

        const int hidden_feature_dims = num_heads * head_dimension;
        auto QKV = Tensor::create({3, total_seq_lens, num_heads, head_dimension}, MemoryType::CPU, DataType::FLOAT16);
        int offset = 0;
        for (int b = 0; b < batch_size; ++b)
        {
            int cu_seq_len = ((int *)(SEQ_LEN->data()))[b + 1] - ((int *)(SEQ_LEN->data()))[b];
            for (int s = 0; s < cu_seq_len; ++s)
            {
                const int offset1 = b * max_seq_length * hidden_feature_dims + s * hidden_feature_dims;
                memcpy((half *)QKV->data() + offset, (half *)Q->data() + offset1, hidden_feature_dims * sizeof(half));
                offset += hidden_feature_dims;
                memcpy((half *)QKV->data() + offset, (half *)K->data() + offset1, hidden_feature_dims * sizeof(half));
                offset += hidden_feature_dims;
                memcpy((half *)QKV->data() + offset, (half *)V->data() + offset1, hidden_feature_dims * sizeof(half));
                offset += hidden_feature_dims;
            }

            const int offset2 = b * max_seq_length * hidden_feature_dims + cu_seq_len * hidden_feature_dims;
            memset((half *)Q->data() + offset2, 0, (max_seq_length - cu_seq_len) * hidden_feature_dims * sizeof(half));
            memset((half *)K->data() + offset2, 0, (max_seq_length - cu_seq_len) * hidden_feature_dims * sizeof(half));
            memset((half *)V->data() + offset2, 0, (max_seq_length - cu_seq_len) * hidden_feature_dims * sizeof(half));
        }

        printf("total_seq_lens: %d\n", total_seq_lens);
        // dump qkv data
        if (0)
        {
            for (int b = 0; b < batch_size; ++b)
            {
                int cu_seq_len = ((int *)(SEQ_LEN->data()))[b + 1] - ((int *)(SEQ_LEN->data()))[b];
                for (int s = 0; s < max_seq_length; ++s)
                {
                    const int offset1 = b * max_seq_length * hidden_feature_dims + s * hidden_feature_dims;

                    // Q
                    for (int l = 0; l < hidden_feature_dims; ++l)
                    {
                        printf("%.4f, ", float(((half *)(Q->data()))[offset1 + l]));
                    }

                    // K
                    printf("\t\t");
                    for (int l = 0; l < hidden_feature_dims; ++l)
                    {
                        printf("%.4f, ", float(((half *)(K->data()))[offset1 + l]));
                    }
                    // V
                    printf("\t\t");
                    for (int l = 0; l < hidden_feature_dims; ++l)
                    {
                        printf("%.4f, ", float(((half *)(V->data()))[offset1 + l]));
                    }
                    printf("\n\n");
                }
            }
        }

        // 填充mask数据
        memset(M->data(), 0, batch_size * max_seq_length * max_seq_length);
        for (int b = 0; b < batch_size; ++b)
        {
            int cu_seq_len = ((int *)(SEQ_LEN->data()))[b + 1] - ((int *)(SEQ_LEN->data()))[b];
            for (int i = 0; i < cu_seq_len; ++i)
            {
                for (int j = 0; j < cu_seq_len; ++j)
                {
                    ((half *)M->data())[b * max_seq_length * max_seq_length + i * cu_seq_len + j] = 1.0f; 
                }
            }
        }

        inputs.emplace_back(Q);
        inputs.emplace_back(K);
        inputs.emplace_back(V);
        inputs.emplace_back(QKV);
        inputs.emplace_back(M);
        inputs.emplace_back(SEQ_LEN);
        inputs.emplace_back(Param);

        auto O_ref = std::make_shared<Tensor>(shape, MemoryType::CPU, DataType::FLOAT16);
        // auto O_ref2 = std::make_shared<Tensor>(shape, MemoryType::CPU, DataType::FLOAT16);
        if (!perf_mode)
        {
            reference_attention(Q, K, V, M, SEQ_LEN, O_ref);
            // reference_attention(QKV, M, SEQ_LEN, O_ref2, batch_size, max_seq_length, num_heads, head_dimension);
        }

        auto O_unfused = Tensor::create(shape, MemoryType::GPU, DataType::FLOAT16);
        outputs_unfused.emplace_back(O_unfused);

        auto O_flash = Tensor::create(shape, MemoryType::GPU, DataType::FLOAT16);
        outputs_flash.emplace_back(O_flash);

        // auto O_byte = Tensor::create(shape, MemoryType::GPU, DataType::FLOAT16);
        // outputs_byte.emplace_back(O_byte);

        Q->toCUDA();
        K->toCUDA();
        V->toCUDA();
        QKV->toCUDA();
        M->toCUDA();
        SEQ_LEN->toCUDA();

        unfused_attention->onResize(inputs, outputs_unfused);
        flash_bert_attention->onResize(inputs, outputs_flash);
        byte_attention->onResize(inputs, outputs_byte);

        runFuncWithTimeProfiler("unfused_attn", run_times, perf_mode, [&] {
            unfused_attention->onExecute(inputs, outputs_unfused);
        });

        runFuncWithTimeProfiler("flash_attn", run_times, perf_mode, [&] {
            flash_bert_attention->onExecute(inputs, outputs_flash);
        });

        // runFuncWithTimeProfiler("byte_attention", run_times, perf_mode, [&] {
        //     byte_attention->onExecute(inputs, outputs_flash2);
        // });

        if (!perf_mode)
        {
            // CE_CHECK(compareTensor(*O_ref, *O_ref2, "qkv_ref"), "compare tensor error");

            O_unfused->toCPU();
            CE_CHECK(compareTensor(*O_ref, *O_unfused, "unfused_attn"), "compare tensor error");

            O_flash->toCPU();
            CE_CHECK(compareTensor(*O_ref, *O_flash, "flash_attn"), "compare tensor error");

            // O_byte->toCPU();
            // CE_CHECK(compareTensor(*O_ref, *O_byte, "byte_attention"), "compare tensor error");
        }

        return true;
    }
};

TestSuiteRegister(AttentionTest, "op/attention");