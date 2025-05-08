#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>

// Block configuration for 2D thread layout
static const int BLOCK_DIM_X = 16;
static const int BLOCK_DIM_Y = 16;
static const int BLOCK_SIZE = BLOCK_DIM_X * BLOCK_DIM_Y;

// Number of streams to use for overlapping kernel execution and memory transfers
static const int N_STREAMS = 4;

// Kernel: processes a segment of the input using a grid-stride loop and 2D thread block reduction in shared memory.
// Each kernel instance writes its partial sum directly into its assigned location in the global partial result array.

template <typename scalar_t>
__global__ void mse_forward_kernel_stream(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ partial_result,  // each launched kernel writes to its own accumulator
    const int64_t start_idx,
    const int64_t seg_size
) {
    // Shared memory for block-level reduction
    __shared__ double sdata[BLOCK_SIZE];

    // 2D thread indexing within the block
    int tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
    
    // Compute the actual index offset for this thread in the segment
    int idx = start_idx + blockIdx.x * BLOCK_SIZE + tid;
    int grid_stride = gridDim.x * BLOCK_SIZE;
    double sum_val = 0.0;

    int end_idx = start_idx + seg_size;
    
    // Grid-stride loop over the assigned segment
    for (; idx < end_idx; idx += grid_stride) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        sum_val += diff * diff;
    }

    // Store the per-thread sum in shared memory
    sdata[tid] = sum_val;
    __syncthreads();

    // Intra-block reduction (binary tree reduction in shared memory)
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // The first thread in the block adds its block's result to the global partial result using atomicAdd.
    if (tid == 0) {
        atomicAdd(partial_result, sdata[0]);
    }
}

// Host forward function: partitions work among multiple CUDA streams to overlap kernel execution and asynchronous memory transfers.
// Each stream launches the kernel on a separate segment of the input, writing to its own slot in the partial results array.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();

    // Determine number of streams to use (at most N_STREAMS, but not more than elements)
    int streams_to_use = N_STREAMS;
    streams_to_use = std::min<int64_t>(streams_to_use, num_elements);
    
    // Partition the input data into chunks, one for each stream
    int64_t chunk_size = (num_elements + streams_to_use - 1) / streams_to_use;

    // Allocate a tensor on device for partial results (one double per stream)
    auto partial_results = torch::zeros({streams_to_use}, predictions.options().dtype(at::kDouble));
    double* d_partial = partial_results.data_ptr<double>();

    // Create CUDA streams
    std::vector<cudaStream_t> streams(streams_to_use);
    for (int i = 0; i < streams_to_use; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Configure 2D blocks
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    int gridSize = 0; // will be computed per chunk

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda_stream_shared", ([&] {
        const scalar_t* preds_ptr = predictions.data_ptr<scalar_t>();
        const scalar_t* tgts_ptr = targets.data_ptr<scalar_t>();
        
        // For each stream, launch the kernel asynchronously on its own data chunk
        for (int i = 0; i < streams_to_use; i++) {
            int64_t start_idx = i * chunk_size;
            int64_t seg_size = std::min(chunk_size, num_elements - start_idx);
            if (seg_size <= 0)
                continue;
            gridSize = (seg_size + BLOCK_SIZE - 1) / BLOCK_SIZE;  // one thread per element in a block
            
            mse_forward_kernel_stream<scalar_t><<<gridSize, block, 0, streams[i]>>>(
                preds_ptr,
                tgts_ptr,
                d_partial + i, // each stream has its dedicated accumulator
                start_idx,
                seg_size
            );
        }
    }));

    // Allocate pinned (page-locked) host memory for asynchronous copy of partial results
    double* h_partial = nullptr;
    cudaHostAlloc((void**)&h_partial, streams_to_use * sizeof(double), cudaHostAllocDefault);
    
    // Asynchronously copy each partial result from device to host
    for (int i = 0; i < streams_to_use; i++) {
        cudaMemcpyAsync(&h_partial[i], d_partial + i, sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Synchronize all streams to ensure completion of kernel execution and memcopies
    for (int i = 0; i < streams_to_use; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    // Sum up the partial results on host to obtain the total sum of squared errors
    double total_sum = 0.0;
    for (int i = 0; i < streams_to_use; i++) {
        total_sum += h_partial[i];
    }
    cudaFreeHost(h_partial);
    
    // Compute the mean squared error
    double mse = total_sum / static_cast<double>(num_elements);
    auto result = torch::full({1}, mse, predictions.options().dtype(at::kDouble));
    result = result.to(predictions.dtype());
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MSE forward with streams and shared memory (CUDA)");
}
