
#include "tensor.hpp"

void flash_decode_attention(Tensor       &q,         // batch_size x seqlen_q x num_heads x head_size
                            const Tensor &k,         // batch_size x seqlen_k x num_heads_k x head_size
                            const Tensor &v,         // batch_size x seqlen_k x num_heads_k x head_size
                            Tensor       &out,       // batch_size x seqlen_q x num_heads x head_size
                            const float   softmax_scale,
                            bool          is_causal);