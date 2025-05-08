#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

static const int BLOCK_SIZE = 256;
static const int N_STREAMS = 4;

// Kernel to compute partial MSE loss over a segment [start_idx, start_idx + seg_size).
// Each kernel launch uses grid-stride looping and reduces using shared memory, then uses atomicAdd to
// accumulate block results into a single double value for the segment.

template <typename scalar_t>
__global__ void mse_forward_kernel_chunk(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ partial_result,  // single double accumulator for this segment
    const int64_t start_idx,
    const int64_t seg_size
) {
    __shared__ double sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    double sum_val = 0.0;
    int grid_stride = blockDim.x * gridDim.x;
    int end_idx = start_idx + seg_size;

    // Grid-stride loop over the segment
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
    // Atomic add block sum into the partial result for this segment
    if (tid == 0) {
        atomicAdd(partial_result, sdata[0]);
    }
}


// Host function: partitions the input into chunks and uses multiple CUDA streams to overlap kernel
// execution with asynchronous memory transfers of the small partial results back to host.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();

    // Decide on the number of streams to use; don't use more streams than elements
    int streams_to_use = N_STREAMS;
    streams_to_use = std::min<int64_t>(streams_to_use, num_elements);
    
    // Determine the chunk size for each stream
    int64_t chunk_size = (num_elements + streams_to_use - 1) / streams_to_use;

    // Allocate a tensor for partial results on device (one double per stream)
    auto partial_results = torch::zeros({streams_to_use}, predictions.options().dtype(at::kDouble));
    double* d_partial = partial_results.data_ptr<double>();

    // Create CUDA streams
    std::vector<cudaStream_t> streams(streams_to_use);
    for (int i = 0; i < streams_to_use; i++) {
        cudaStreamCreate(&streams[i]);
    }

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda_stream", ([&] {
        const scalar_t* preds_ptr = predictions.data_ptr<scalar_t>();
        const scalar_t* tgts_ptr = targets.data_ptr<scalar_t>();
        
        // Launch a kernel for each chunk in its own stream
        for (int i = 0; i < streams_to_use; i++) {
            int64_t start_idx = i * chunk_size;
            int64_t seg_size = std::min(chunk_size, num_elements - start_idx);
            if (seg_size <= 0) continue;
            int grid_size = (seg_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            // Launch the kernel asynchronously on stream[i]
            mse_forward_kernel_chunk<scalar_t><<<grid_size, BLOCK_SIZE, 0, streams[i]>>>(
                preds_ptr,
                tgts_ptr,
                d_partial + i,  // each stream writes to its own accumulator slot
                start_idx,
                seg_size
            );
        }
    }));

    // Allocate pinned (page-locked) host memory for asynchronous copy of partial results
    double* h_partial = nullptr;
    cudaHostAlloc((void**)&h_partial, streams_to_use * sizeof(double), cudaHostAllocDefault);
    
    // Asynchronously copy each partial result from device to host using its stream
    for (int i = 0; i < streams_to_use; i++) {
        cudaMemcpyAsync(&h_partial[i], d_partial + i, sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Synchronize all streams to ensure kernel execution and memcopies are complete
    for (int i = 0; i < streams_to_use; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Sum the partial results on host to get the total sum of squared errors
    double total_sum = 0.0;
    for (int i = 0; i < streams_to_use; i++) {
        total_sum += h_partial[i];
    }
    cudaFreeHost(h_partial);

    double mse = total_sum / static_cast<double>(num_elements);
    
    // Create the final result tensor (using double for precision) and cast to the predictions dtype
    auto result = torch::full({1}, mse, predictions.options().dtype(at::kDouble));
    result = result.to(predictions.dtype());
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward with overlapped streams (CUDA)");
}
