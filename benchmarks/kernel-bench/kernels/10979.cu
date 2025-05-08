#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 256
#define NUM_STREAMS 4
#define CHUNK_SIZE (1 << 20)  // 1M elements per chunk

template <typename scalar_t>
__global__ void mse_hybrid_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ partial_sums,
    const int64_t start,
    const int64_t chunk_length,
    const int stream_idx
) {
    __shared__ double shm[BLOCK_SIZE];
    const int tid = threadIdx.x;
    int global_idx = start + blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    double thread_sum = 0.0;
    
    #pragma unroll 4
    for (int idx = global_idx; idx < start + chunk_length; idx += stride) {
        if (idx < start + chunk_length) {
            const double pred = static_cast<double>(preds[idx]);
            const double tgt = static_cast<double>(tgts[idx]);
            const double diff = pred - tgt;
            thread_sum += diff * diff;
        }
    }
    
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    if (tid % warpSize == 0) {
        shm[tid/warpSize] = thread_sum;
    }
    __syncthreads();
    
    if (tid < warpSize) {
        thread_sum = (tid < blockDim.x/warpSize) ? shm[tid] : 0.0;
        
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        if (tid == 0) {
            atomicAdd(&partial_sums[stream_idx], thread_sum);
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    
    auto partial_results = torch::zeros({NUM_STREAMS}, 
                                      predictions.options().dtype(at::kDouble));
    double* d_partial = partial_results.data_ptr<double>();
    
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    const int64_t elements_per_stream = (num_elements + NUM_STREAMS - 1) / NUM_STREAMS;
    const int64_t chunks_per_stream = (elements_per_stream + CHUNK_SIZE - 1) / CHUNK_SIZE;
    
    for (int stream_idx = 0; stream_idx < NUM_STREAMS; stream_idx++) {
        const int64_t stream_start = stream_idx * elements_per_stream;
        
        for (int chunk = 0; chunk < chunks_per_stream; chunk++) {
            const int64_t chunk_start = stream_start + chunk * CHUNK_SIZE;
            if (chunk_start >= num_elements) continue;
            
            const int64_t chunk_length = std::min(CHUNK_SIZE, 
                                                num_elements - chunk_start);
            const int grid_size = (chunk_length + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_hybrid_cuda", ([&] {
                mse_hybrid_kernel<scalar_t><<<grid_size, BLOCK_SIZE, 0, streams[stream_idx]>>>(
                    predictions.data_ptr<scalar_t>(),
                    targets.data_ptr<scalar_t>(),
                    d_partial,
                    chunk_start,
                    chunk_length,
                    stream_idx
                );
            }));
        }
    }

    auto result = partial_results.sum().div_(static_cast<double>(num_elements));
    
    for (auto& stream : streams) {
        cudaStreamDestroy(stream);
    }

    return result.to(predictions.dtype());
}