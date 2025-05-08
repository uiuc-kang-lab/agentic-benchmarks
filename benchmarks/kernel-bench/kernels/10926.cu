#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <cmath>
#include <algorithm>

// Define block dimensions for 2D thread block and number of streams
static const int BLOCK_DIM_X = 16;
static const int BLOCK_DIM_Y = 16;  // Total threads per block = 256
static const int N_STREAMS = 4;

// Kernel: Compute MSE over a segment of the input using a grid-stride loop and intra-block reduction
// using a 2D block configuration. Each kernel instance computes a partial sum for its assigned segment
// and writes the result into a dedicated accumulator via atomicAdd.

template <typename scalar_t>
__global__ void mse_forward_kernel_chunk(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ partial_result,  // Each kernel writes its partial sum here
    const int64_t start_idx,              // Starting index for the current segment/chunk
    const int64_t seg_size                // Number of elements in this segment
) {
    // Shared memory for intra-block reduction
    __shared__ double shm[BLOCK_DIM_X * BLOCK_DIM_Y];

    const int num_threads = BLOCK_DIM_X * BLOCK_DIM_Y;
    // Flatten the 2D thread index into 1D
    int tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;

    // Linear block index computed from a 2D grid
    int bid = blockIdx.y * gridDim.x + blockIdx.x;
    
    // Compute the initial global index for this thread in the segment
    int64_t index = start_idx + bid * num_threads + tid;
    
    // Total number of threads participating across the grid for this kernel launch
    int grid_stride = num_threads * gridDim.x * gridDim.y;

    double thread_sum = 0.0;
    int64_t end_idx = start_idx + seg_size;

    // Grid-stride loop over the assigned segment
    for (; index < end_idx; index += grid_stride) {
        double diff = static_cast<double>(preds[index]) - static_cast<double>(tgts[index]);
        thread_sum += diff * diff;
    }

    // Store each thread's partial sum in shared memory
    shm[tid] = thread_sum;
    __syncthreads();

    // Intra-block reduction (binary tree reduction)
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shm[tid] += shm[tid + s];
        }
        __syncthreads();
    }

    // The first thread in the block atomically adds the block's result to the partial result
    if (tid == 0) {
        atomicAdd(partial_result, shm[0]);
    }
}

// Host function: partitions work among multiple CUDA streams and launches the kernel for each segment

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();

    // Decide on number of streams (no more than N_STREAMS and not more than num_elements)
    int num_streams = std::min((int64_t)N_STREAMS, num_elements);
    // Partition the data into approximately equal chunks
    int64_t chunk_size = (num_elements + num_streams - 1) / num_streams;

    // Allocate a tensor for partial results (one double per stream) on the device
    auto partial_results = torch::zeros({num_streams}, predictions.options().dtype(at::kDouble));
    double* d_partial = partial_results.data_ptr<double>();

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch the kernel for each chunk on its own stream using AT_DISPATCH for the tensor's scalar type
    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda_combined", ([&] {
        const scalar_t* preds_ptr = predictions.data_ptr<scalar_t>();
        const scalar_t* tgts_ptr = targets.data_ptr<scalar_t>();

        for (int i = 0; i < num_streams; i++) {
            int64_t start_idx = i * chunk_size;
            int64_t seg_size = std::min(chunk_size, num_elements - start_idx);
            if (seg_size <= 0) continue;

            // Compute the total number of blocks needed given the fixed block size
            int threads_per_block = BLOCK_DIM_X * BLOCK_DIM_Y;
            int num_blocks = (seg_size + threads_per_block - 1) / threads_per_block;
            // Use a 2D grid configuration for better occupancy
            int grid_dim_x = std::max(1, (int)std::ceil(std::sqrt((double)num_blocks)));
            int grid_dim_y = grid_dim_x;
            dim3 grid_dim(grid_dim_x, grid_dim_y);
            dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);

            // Launch the kernel asynchronously on stream[i]
            mse_forward_kernel_chunk<scalar_t><<<grid_dim, block_dim, 0, streams[i]>>>(
                preds_ptr,
                tgts_ptr,
                d_partial + i,  // Each stream writes to a unique accumulator slot
                start_idx,
                seg_size
            );
        }
    }));

    // Allocate pinned (page-locked) host memory for asynchronous partial result transfer
    double* h_partial = nullptr;
    cudaHostAlloc((void**)&h_partial, num_streams * sizeof(double), cudaHostAllocDefault);
    
    // Asynchronously copy each partial result from device to host using its respective stream
    for (int i = 0; i < num_streams; i++) {
        cudaMemcpyAsync(&h_partial[i], d_partial + i, sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize all streams to ensure complete execution and memory transfers
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Aggregate the partial results on the host
    double total_sum = 0.0;
    for (int i = 0; i < num_streams; i++) {
        total_sum += h_partial[i];
    }
    cudaFreeHost(h_partial);

    // Compute the final mean squared error
    double mse = total_sum / static_cast<double>(num_elements);
    auto result = torch::full({1}, mse, predictions.options().dtype(at::kDouble));
    result = result.to(predictions.dtype());
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined MSE forward with multi-stream and 2D reduction (CUDA)");
}
