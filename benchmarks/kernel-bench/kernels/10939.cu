/*
This CUDA extension computes the Mean Squared Error (MSE) by splitting the input into chunks,
launching a kernel on multiple streams to compute partial sums, and then reducing these partial results
on the GPU. This removes the need to asynchronously copy small partials back to host memory.
*/

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Configuration constants
#define BLOCK_SIZE 256
#define N_STREAMS 4

// Kernel: Compute partial MSE sum over a segment specified by [start_idx, start_idx + seg_size).
// Each block reduces its own share using shared memory, then atomically adds its result into the
// partial result accumulator for that segment.

template <typename scalar_t>
__global__ void mse_chunk_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ partial_result,
    const int64_t start_idx,
    const int64_t seg_size
) {
    __shared__ double sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    double sum_val = 0.0;
    int grid_stride = blockDim.x * gridDim.x;
    int end_idx = start_idx + seg_size;

    // Grid-stride loop over the assigned segment
    for (; idx < end_idx; idx += grid_stride) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        sum_val += diff * diff;
    }

    sdata[tid] = sum_val;
    __syncthreads();

    // Intra-block reduction in shared memory
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Atomic add the block's sum into the segment's partial result
    if (tid == 0) {
        atomicAdd(partial_result, sdata[0]);
    }
}

// Kernel: Sum an array of partial sums into a single result.
// This is a simple parallel reduction kernel launched using one block.

__global__ void final_reduce_kernel(
    const double* __restrict__ partial_sums,
    int n,
    double* __restrict__ result
) {
    __shared__ double sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    double sum = 0.0;

    // Each thread sums over multiple elements if needed
    for (int i = tid; i < n; i += blockDim.x) {
         sum += partial_sums[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
         if (tid < s) {
             sdata[tid] += sdata[tid + s];
         }
         __syncthreads();
    }

    if (tid == 0) {
        *result = sdata[0];
    }
}

// Helper function to launch the mse_chunk_kernel
template <typename scalar_t>
void launch_mse_chunk_kernel(
    const scalar_t* preds,
    const scalar_t* tgts,
    double* partial_result,
    int64_t start_idx,
    int64_t seg_size,
    cudaStream_t stream
) {
    if (seg_size <= 0) return;
    int grid_size = (seg_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    mse_chunk_kernel<scalar_t><<<grid_size, BLOCK_SIZE, 0, stream>>>(
         preds, tgts, partial_result, start_idx, seg_size);
}

// The forward function partitions the work among multiple CUDA streams,
// each computing a partial MSE sum. A final reduction kernel then remotely sums
// these partials to yield the total squared error, which is divided by num_elements
// to produce the MSE.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();

    // Decide on the number of streams (chunks) to use
    int streams_to_use = N_STREAMS;
    streams_to_use = std::min<int64_t>(streams_to_use, num_elements);
    int64_t chunk_size = (num_elements + streams_to_use - 1) / streams_to_use;

    // Allocate a device tensor for partial results (one double per stream)
    auto partial_results = torch::zeros({streams_to_use}, predictions.options().dtype(at::kDouble));
    double* d_partial = partial_results.data_ptr<double>();

    // Create CUDA streams
    std::vector<cudaStream_t> streams(streams_to_use);
    for (int i = 0; i < streams_to_use; i++) {
         cudaStreamCreate(&streams[i]);
         // partial_results is already zeroed
    }

    // Launch a kernel for each chunk on its own stream
    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda_combined", ([&] {
         const scalar_t* preds_ptr = predictions.data_ptr<scalar_t>();
         const scalar_t* tgts_ptr = targets.data_ptr<scalar_t>();
         for (int i = 0; i < streams_to_use; i++) {
             int64_t start_idx = i * chunk_size;
             int64_t seg_size = std::min(chunk_size, num_elements - start_idx);
             if (seg_size <= 0) continue;
             launch_mse_chunk_kernel<scalar_t>(
                  preds_ptr,
                  tgts_ptr,
                  d_partial + i,  // each stream writes to its unique slot
                  start_idx,
                  seg_size,
                  streams[i]
             );
         }
    }));

    // Synchronize and destroy streams
    for (int i = 0; i < streams_to_use; i++) {
         cudaStreamSynchronize(streams[i]);
         cudaStreamDestroy(streams[i]);
    }

    // Allocate a device tensor for the final reduction result
    auto total_result = torch::empty({1}, predictions.options().dtype(at::kDouble));
    double* d_total = total_result.data_ptr<double>();

    // Launch the final reduction kernel (using one block)
    final_reduce_kernel<<<1, BLOCK_SIZE>>>(d_partial, streams_to_use, d_total);
    cudaDeviceSynchronize();

    // Divide by the total number of elements to get the mean squared error
    double mse = (*d_total) / static_cast<double>(num_elements);
    
    // Create a result tensor with the MSE, cast to the original dtype
    auto result = torch::full({1}, mse, predictions.options().dtype(at::kDouble));
    result = result.to(predictions.dtype());
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("forward", &forward, "Mean Squared Error forward with Streams and GPU Reduction (CUDA)");
}
