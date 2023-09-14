#include "flash_attention.hpp"
#include "tensor.hpp"

// Tensor Shape[b, s, h, d]
// 如下函数知识为了考虑对FlashAttention手动进行实现，特化了b=x, s=128 * n, h=32, d=128版本
// 纯属学习使用，不用于实际的模型推理，q, k, v考虑大小一致，也就是只考虑mha，不考虑gqa, mqa
std::vector<Tensor> flash_attention_mha_bxsxh32d128(const Tensor& q,     // batch_size x seq_len x num_heads x head_size
                                                    const Tensor& k,     // batch_size x seq_len x num_heads x head_size
                                                    const Tensor& v,     // batch_size x seq_len x num_heads x head_size
                                                    Tensor&       out,   // batch_size x seq_len x num_heads x head_size
                                                    const int     max_seqlen,
                                                    const float   softmax_scale,
                                                    const bool    zero_tensors,
                                                    const bool    is_causal,
                                                    const int     num_splits)
{
    const int b  = q.batch();    // batch size
    const int s  = q.seqlen();   // sequence len
    const int h  = q.numheads(); // num heads
    const int d  = q.headdim();  // head dims
    CE_CHECK((s % 128 == 0), "seq length能被128整除");
    CE_CHECK((b >= 1), "batch size 参数检查");
    CE_CHECK((h == 32), "head numbers should be equal 32");
    CE_CHECK((d == 128), "head dimension should be equal 128");

    // run_flash_fwd();

    return {};
}