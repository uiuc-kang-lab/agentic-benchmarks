#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;
static const int NUM_STREAMS = 4;

template <typename scalar_t>
__global__ void mse_forward_kernel_streamed(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements,
    const int64_t chunk_size,
    const int64_t chunk_offset
) {
    __shared__ double shm[BLOCK_SIZE];
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int64_t start_idx = chunk_offset;
    const int64_t end_idx = min(start_idx + chunk_size, num_elements);
    
    double thread_sum = 0.0;
    
    // Process chunk of data
    for(int64_t idx = start_idx + gid; idx < end_idx; idx += blockDim.x * gridDim.x) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
    }
    
    // Shared memory reduction
    shm[tid] = thread_sum;
    __syncthreads();
    
    // Reduce within block
    for(int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if(tid < stride) {
            shm[tid] += shm[tid + stride];
        }
        __syncthreads();
    }
    
    if(tid == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for(int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Calculate chunk size for each stream
    const int64_t chunk_size = (num_elements + NUM_STREAMS - 1) / NUM_STREAMS;
    const int blocks_per_chunk = (chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        for(int i = 0; i < NUM_STREAMS; i++) {
            const int64_t chunk_offset = i * chunk_size;
            mse_forward_kernel_streamed<scalar_t><<<blocks_per_chunk, BLOCK_SIZE, 0, streams[i]>>>(
                predictions.data_ptr<scalar_t>(),
                targets.data_ptr<scalar_t>(),
                accumulator.data_ptr<double>(),
                num_elements,
                chunk_size,
                chunk_offset
            );
        }
    });

    // Synchronize and cleanup streams
    for(int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward with streams (CUDA)");
}