#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_<cutlass::half_t, 64>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64<cutlass::half_t>(params, stream);
}

template<>
void run_mha_fwd_<cutlass::half_t, 128>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128<cutlass::half_t>(params, stream);
}


template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 64>(Flash_fwd_params &params, cudaStream_t stream);
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128>(Flash_fwd_params &params, cudaStream_t stream);