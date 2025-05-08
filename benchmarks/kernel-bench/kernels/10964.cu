#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define block size and number of streams for pipelining
#define BLOCK_SIZE 256
#define NUM_STREAMS 4

// Kernel to compute partial sum for a chunk of data
// Each kernel processes elements in the range [start, start + chunk_length)
// and accumulates the squared differences into a single double value using shared memory reduction.

template <typename scalar_t>
__global__ void mse_chunk_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ partial_sum,
    const int64_t start,
    const int64_t chunk_length
) {
    __shared__ double shm[BLOCK_SIZE];
    int tid = threadIdx.x;
    // Calculate global index offset for the chunk
    int global_idx = start + blockIdx.x * blockDim.x + tid;
    double thread_sum = 0.0;
    int stride = blockDim.x * gridDim.x;
    
    // Process the assigned chunk using a grid-stride loop
    for (int idx = global_idx; (idx - start) < chunk_length; idx += stride) {
        // Check bounds for safety
        if (idx < start + chunk_length) {
            double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
            thread_sum += diff * diff;
        }
    }
    
    // Perform reduction in shared memory
    shm[tid] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shm[tid] += shm[tid + s];
        }
        __syncthreads();
    }
    
    // The first thread in the block atomically adds the block result to the chunk's partial sum
    if (tid == 0) {
        atomicAdd(partial_sum, shm[0]);
    }
}

// Host function that partitions the input and uses multiple CUDA streams to overlap computation with memory transfers

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();

    // Create a device tensor to hold partial results (one per stream)
    auto partial_results = torch::zeros({NUM_STREAMS}, predictions.options().dtype(at::kDouble));
    double* d_partial = partial_results.data_ptr<double>();

    // Allocate pinned host memory for asynchronous copy of partial results
    double* h_partial = nullptr;
    cudaMallocHost(&h_partial, NUM_STREAMS * sizeof(double));

    // Create CUDA streams
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Partition the data into chunks for each stream
    int64_t chunk_size = (num_elements + NUM_STREAMS - 1) / NUM_STREAMS;

    // Launch kernels concurrently in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int64_t start = i * chunk_size;
        int64_t current_chunk = ((start + chunk_size) < num_elements) ? chunk_size : (num_elements - start);
        
        // If no elements are assigned to this stream, set the partial result to 0
        if (current_chunk <= 0) {
            cudaMemsetAsync(&d_partial[i], 0, sizeof(double), streams[i]);
            continue;
        }

        int grid_size = (current_chunk + BLOCK_SIZE - 1) / BLOCK_SIZE; grid_size = min(grid_size, (chunk_length + BLOCK_SIZE - 1) / BLOCK_SIZE);

        AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_chunk_cuda", ([&] {
            mse_chunk_kernel<scalar_t><<<grid_size, BLOCK_SIZE, 0, streams[i]>>>(
                predictions.data_ptr<scalar_t>(),
                targets.data_ptr<scalar_t>(),
                &d_partial[i],
                start,
                current_chunk
            );
        }));

        // Asynchronously copy the partial result from device to host using the same stream
        cudaMemcpyAsync(&h_partial[i], &d_partial[i], sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize all streams to ensure kernels and copies are complete
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Sum up the partial results on the host
    double total_sum = 0.0;
    for (int i = 0; i < NUM_STREAMS; i++) {
        total_sum += h_partial[i];
    }
    cudaFreeHost(h_partial);

    // Compute the mean squared error
    total_sum /= static_cast<double>(num_elements);

    // Return the result converted to the predictions' data type
    auto result = torch::tensor({total_sum}, predictions.options().dtype(at::kDouble));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward using CUDA streams for overlap");
}
