#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 256
#define NUM_STREAMS 4
#define ELEMENTS_PER_THREAD 4  // Process multiple elements per thread

template <typename scalar_t>
__global__ void mse_hybrid_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ partial_sum,
    const int64_t start,
    const int64_t chunk_length
) {
    __shared__ double shm[BLOCK_SIZE];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    double thread_sum = 0.0;
    
    // Process multiple elements per thread using loop unrolling
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int64_t idx = start + gid * ELEMENTS_PER_THREAD + i;
        if (idx < start + chunk_length) {
            const double pred = static_cast<double>(preds[idx]);
            const double tgt = static_cast<double>(tgts[idx]);
            const double diff = pred - tgt;
            thread_sum += diff * diff;
        }
    }
    
    // Warp-level reduction first (assumes warp size of 32)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // Block-level reduction using shared memory
    if (tid % 32 == 0) {
        shm[tid / 32] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (tid < (BLOCK_SIZE / 32)) {
        double warp_sum = shm[tid];
        #pragma unroll
        for (int offset = (BLOCK_SIZE / 64); offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (tid == 0) {
            atomicAdd(partial_sum, warp_sum);
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto partial_results = torch::zeros({NUM_STREAMS}, predictions.options().dtype(at::kDouble));
    double* d_partial = partial_results.data_ptr<double>();

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int64_t chunk_size = (num_elements + NUM_STREAMS - 1) / NUM_STREAMS;
    const int elements_per_block = BLOCK_SIZE * ELEMENTS_PER_THREAD;

    for (int i = 0; i < NUM_STREAMS; i++) {
        const int64_t start = i * chunk_size;
        const int64_t current_chunk = std::min(chunk_size, num_elements - start);
        
        if (current_chunk <= 0) continue;

        const int grid_size = (current_chunk + elements_per_block - 1) / elements_per_block;

        AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_hybrid_cuda", ([&] {
            mse_hybrid_kernel<scalar_t><<<grid_size, BLOCK_SIZE, 0, streams[i]>>>(
                predictions.data_ptr<scalar_t>(),
                targets.data_ptr<scalar_t>(),
                &d_partial[i],
                start,
                current_chunk
            );
        }));
    }

    // Sum partial results on GPU instead of CPU
    auto final_sum = partial_results.sum();
    auto result = final_sum.div_(static_cast<double>(num_elements));

    // Cleanup streams
    for (auto& stream : streams) {
        cudaStreamDestroy(stream);
    }

    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized MSE forward (CUDA)");
}