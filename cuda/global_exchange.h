#include "stream_manager.h"
#ifdef FMOE_USE_NCCL

void fmoe_cuda_expert_exchange_impl(
        const int* local_expert_count, 
        int* global_expert_count, 
        int n_expert, int world_size,
        CudaStreamManager* smgr) {
    NCCL_SAFE_CALL(ncclGroupStart());
    for (int i = 0; i < world_size; ++i) {
        NCCL_SAFE_CALL(ncclSend(
                local_expert_count + n_expert * i,
                n_expert,
                ncclInt32,
                i,
                smgr->ncclcomm[0],
                smgr->stream(0)));
        NCCL_SAFE_CALL(ncclRecv(
                global_expert_count + n_expert * i,
                n_expert,
                ncclInt32,
                i,
                smgr->ncclcomm[0],
                smgr->stream(0)));
    }
    NCCL_SAFE_CALL(ncclGroupEnd());
    smgr->sync(1);
}

template<typename scalar_t>
void fmoe_cuda_global_scatter_impl(
    const scalar_t* local_input_buf,
    const int* local_expert_count,
    const int* global_expert_count,
    scalar_t* input_buf,
    int in_feat, int n_expert, int world_size,
    CudaStreamManager* smgr) {
    // assert world_size > 1
    int recv_ptr = 0;
    /* TODO: may save for backward */
    int*expert_ptr = new int[n_expert * world_size];
    expert_ptr[0] = 0;
    for (int i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    for (int i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (int j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        local_input_buf + expert_ptr[idx] * in_feat, 
                        local_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar, 
                        j,
                        smgr->ncclcomm[i % 2],
                        smgr->stream(i % 2)));
            }
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        input_buf + recv_ptr * in_feat,
                        global_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm[i % 2],
                        smgr->stream(i % 2)));
                recv_ptr += global_expert_count[idx];
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
    smgr->sync(2);
}

template<typename scalar_t>
void fmoe_cuda_global_gather_impl(
    const scalar_t* output_buf,
    const int* local_expert_count,
    const int* global_expert_count,
    scalar_t* local_output_buf,
    int out_feat, int n_expert, int world_size,
    CudaStreamManager* smgr) {
    int send_ptr = 0;
    /* TODO: may save for backward */
    int *expert_ptr = new int[n_expert * world_size];
    expert_ptr[0] = 0;
    for (int i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    for (int i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (int j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        output_buf + send_ptr * out_feat,
                        global_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm[i % 2],
                        smgr->stream(i % 2)));
                send_ptr += global_expert_count[idx];
            }
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        local_output_buf + expert_ptr[idx] * out_feat, 
                        local_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar, 
                        j,
                        smgr->ncclcomm[i % 2],
                        smgr->stream(i % 2)));
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
    smgr->sync(2);
}


#endif  // FMOE_USE_NCCL
