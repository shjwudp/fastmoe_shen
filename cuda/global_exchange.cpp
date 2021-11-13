#include "global_exchange.h"
#include "utils/fmoe_utils.h"
#include <torch/extension.h>
#include <cuda_fp16.h>

#ifdef FMOE_USE_NCCL
#include <nccl.h>

extern "C" size_t array_min_max_size_f16_host(
    at::Half *input, 
    int input_num_element,
    at::Half *output,
    cudaStream_t stream);

extern "C" void compress_f16_to_uint8_host(
    at::Half *input, 
    int input_num_element, 
    int chunk_size,
    int num_chunks,
    uint8_t *output,
    size_t output_size,
    void *dev_buffer,
    size_t dev_size,
    int target_chunk,
    cudaStream_t stream);

extern "C" void decompress_uint8_to_f16_host(
    uint8_t *input,
    size_t input_size,
    int chunk_size,
    int num_chunks,
    at::Half *output,
    cudaStream_t stream);

int align_size(int size, int align) {
  return ((size) + (align) - 1) / (align) * (align);
}

int get_compressed_buffer_size(int n_chunks, int chunk_size) {
  int align_bytes = 32;
  int compressed_align_bytes = align_size(chunk_size, align_bytes) * n_chunks;
  int min_max_align_bytes = align_size(2 * 2, align_bytes) * n_chunks;
  return compressed_align_bytes + min_max_align_bytes;
}


torch::Tensor _expert_exchange(
        torch::Tensor local_expert_count,
        long n_expert, long n_workers) {
    auto global_expert_count = torch::empty_like(local_expert_count);
    auto smgr = getCudaStreamManager(local_expert_count.device().index());

    fmoe_cuda_expert_exchange_impl(
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            n_expert, n_workers,
            smgr);
    return global_expert_count;
}

torch::Tensor _global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers) {
    CHECK_INPUT(input_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto in_feat = input_buf.size(1);
    auto global_input_buf = input_buf.new_empty({batch_size, in_feat});
    auto smgr = getCudaStreamManager(input_buf.device().index());

    // 0  428.9
    long* expert_ptr = new long[local_expert_count.size(0)];
    long* local_expert_compress_ptr = new long[local_expert_count.size(0)];
    long* local_expert_count_ptr = local_expert_count.data_ptr<long>();
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * n_workers; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count_ptr[i - 1];
    }

    long* global_expert_count_ptr = global_expert_count.data_ptr<long>();
    long* global_expert_compress_ptr = new long[local_expert_count.size(0)];
    long local_compress_size = 0;
    long global_compress_size = 0;
    for (size_t j = 0; j < n_workers; ++j) {
      for (size_t i = 0; i < n_expert; ++i) {
            int idx = i + j * n_expert;
            if (local_expert_count_ptr[idx]) {
              local_expert_compress_ptr[idx] = get_compressed_buffer_size(1, local_expert_count_ptr[idx] * in_feat);
              local_compress_size += local_expert_compress_ptr[idx];
            } else {
              local_expert_compress_ptr[idx] = 0;
            }
            if (global_expert_count_ptr[idx]) {
              global_expert_compress_ptr[idx] = get_compressed_buffer_size(1, global_expert_count_ptr[idx] * in_feat);
              global_compress_size += global_expert_compress_ptr[idx];
            } else {
              global_expert_compress_ptr[idx] = 0;
            }
        }
    }
    // 1 417.2 
    auto local_compressed = input_buf.new_empty({local_compress_size}, at::ScalarType::Byte);
    auto global_compressed = input_buf.new_empty({global_compress_size}, at::ScalarType::Byte);

    size_t temp_buff_size = array_min_max_size_f16_host(
        input_buf.data_ptr<at::Half>(),
        input_buf.numel(),
        reinterpret_cast<at::Half*>(local_compressed.data_ptr<uint8_t>()),
        smgr->stream(0));
    auto temp_buff = input_buf.new_empty({temp_buff_size}, at::ScalarType::Byte);

    // 2 493.8
    long local_compress_ptr = 0;
    for (size_t j = 0; j < n_workers; ++j) {
      for (size_t i = 0; i < n_expert; ++i) {
        int idx = i + j * n_expert;
        if (local_expert_count_ptr[idx]) {
          compress_f16_to_uint8_host(
              input_buf.data_ptr<at::Half>() + expert_ptr[idx] * in_feat,
              local_expert_count_ptr[idx] * in_feat,
              local_expert_count_ptr[idx] * in_feat,
              1,
              local_compressed.data_ptr<uint8_t>() + local_compress_ptr,
              local_expert_compress_ptr[idx],
              temp_buff.data_ptr<uint8_t>(),
              temp_buff_size,
              -1,
              smgr->stream(0));
          local_compress_ptr += local_expert_compress_ptr[idx];
        }
      }
    }

    //AT_DISPATCH_INTEGRAL_TYPES(local_compressed.scalar_type(),
    //        "fmoe_cuda_global_scatter", ([&] {
    //    fmoe_cuda_global_scatter_impl<scalar_t>(
    //        local_compressed.data_ptr<scalar_t>(),
    //        local_expert_compress_ptr,
    //        global_expert_compress_ptr,
    //        global_compressed.data_ptr<scalar_t>(),
    //        1, n_expert, n_workers,
    //        smgr
    //    );
    //}));

    // 3 498.8/499.7 nocompress 444.6
    long global_compress_ptr = 0;
    long global_ptr = 0;
    for (size_t i = 0; i < n_expert; ++i) {
      for (size_t j = 0; j < n_workers; ++j) {
        int idx = i + j * n_expert;
        if (global_expert_count_ptr[idx]) {
          decompress_uint8_to_f16_host(
              global_compressed.data_ptr<uint8_t>() + global_compress_ptr,
              global_expert_compress_ptr[idx],
              global_expert_count_ptr[idx] * in_feat,
              1,
              global_input_buf.data_ptr<at::Half>() + global_ptr,
              smgr->stream(0));
          global_ptr += global_expert_count_ptr[idx] * in_feat;
          global_compress_ptr += global_expert_compress_ptr[idx];
        }
     }
    }

    delete[] expert_ptr;
    delete[] local_expert_compress_ptr;
    delete[] global_expert_compress_ptr;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(), 
            "fmoe_cuda_global_scatter", ([&] {
        fmoe_cuda_global_scatter_impl<scalar_t>(
        input_buf.data_ptr<scalar_t>(),
        local_expert_count.data_ptr<long>(),
        global_expert_count.data_ptr<long>(),
        global_input_buf.data_ptr<scalar_t>(),
        in_feat, n_expert, n_workers,
        smgr
        );
     }));
    return global_input_buf;
}

torch::Tensor _global_gather(
        torch::Tensor compressed,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers) {
    CHECK_INPUT(compressed);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto out_feat = compressed.size(1);
    auto local_compressed = compressed.new_empty({batch_size, out_feat});
    auto smgr = getCudaStreamManager(compressed.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(compressed.scalar_type(), 
            "fmoe_cuda_global_gather", ([&] {
        fmoe_cuda_global_gather_impl<scalar_t>(
            compressed.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            local_compressed.data_ptr<scalar_t>(),
            out_feat, n_expert, n_workers,
            smgr
        );
    }));
    return local_compressed;
}

#include <c10d/ProcessGroupNCCL.hpp>

class HackNCCLGroup: public c10d::ProcessGroupNCCL {
public:
    ncclComm_t getcomm(at::Device dev) {
        ncclUniqueId ncclID;
        int rank = getRank();
        if (rank == 0) {
            ncclGetUniqueId(&ncclID);
        }
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8))
        broadcastUniqueNCCLID(&ncclID,
                c10d::OpType::SEND,
                "fastmoe_nccl_comm",
                rank);
#else
        broadcastUniqueNCCLID(&ncclID);
#endif
        ncclComm_t comm;
        NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
        return comm;
    }
};

void _ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t) {
    auto smgr = getCudaStreamManager(t.device().index());
    if (smgr->ncclgood) {
        return;
    }
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)&p;
    smgr->ncclcomm = h->getcomm(t.device());
    if (smgr->ncclcomm != 0) {
        smgr->ncclgood = 1;
    } else {
        std::cerr << "Nccl initialization failed\n";
    }
}

#endif  // FMOE_USE_NCCL
