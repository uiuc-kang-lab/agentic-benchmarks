#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel to compute partial KL divergence for a chunk of the data
// Processes elements in the range [start, start + len) from the input arrays

__global__ void kldiv_kernel_chunk(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ chunk_result,
    int start, 
    int len) {

    const unsigned int warp_size = 32;
    unsigned int lane_id = threadIdx.x % warp_size;
    unsigned int warp_id = threadIdx.x / warp_size;

    extern __shared__ float sdata[]; // Shared memory for warp-level sums
    float sum = 0.0f;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process elements with grid stride; index relative to chunk start
    for (int i = idx; i < len; i += stride) {
        int global_i = start + i;
        float log_pred = log_predictions[global_i];
        float target = targets[global_i];
        sum += expf(log_pred) - target * log_pred;
    }

    // Warp-level reduction using shuffle instructions
    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Each warp's first lane writes its result to shared memory
    if (lane_id == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction by the first warp of the block
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x / warp_size)) ? sdata[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane_id == 0) {
            atomicAdd(chunk_result, sum);
        }
    }
}

// Forward function that splits the input into chunks and uses multiple CUDA streams
// to overlap kernel execution and memory operations

torch::Tensor kldiv_async_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    // Ensure inputs are contiguous
    log_predictions = log_predictions.contiguous();
    targets = targets.contiguous();

    const int n = log_predictions.numel();
    
    // Decide on number of chunks/streams to use (e.g., 4 streams)
    int num_chunks = 4;
    int chunk_size = (n + num_chunks - 1) / num_chunks;  // ceiling division

    // Allocate a tensor on the device to hold partial results from each chunk
    auto options = log_predictions.options();
    torch::Tensor partial_results = torch::zeros({num_chunks}, options);

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_chunks);
    for (int i = 0; i < num_chunks; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Kernel launch configuration
    const int threads_per_block = 256;
    int shared_mem = (threads_per_block / 32) * sizeof(float);

    // Launch a kernel for each chunk on its dedicated stream
    for (int i = 0; i < num_chunks; ++i) {
        int start = i * chunk_size;
        int len = std::min(chunk_size, n - start);
        if (len <= 0) break; // if no elements remain, exit loop
        int blocks = (len + threads_per_block - 1) / threads_per_block;

        // Launch kernel asynchronously on stream i
        kldiv_kernel_chunk<<<blocks, threads_per_block, shared_mem, streams[i]>>>(
            log_predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            partial_results.data_ptr<float>() + i,
            start,
            len
        );
    }

    // Synchronize all streams and destroy them
    for (int i = 0; i < num_chunks; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    // Sum the partial results (can be performed on the GPU as it is a small array)
    torch::Tensor total = partial_results.sum();

    // Return the average KL divergence, ensuring correct result by dividing with n
    return total / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kldiv_async_cuda_forward, "Asynchronous KL divergence forward (CUDA) using streams");
}
