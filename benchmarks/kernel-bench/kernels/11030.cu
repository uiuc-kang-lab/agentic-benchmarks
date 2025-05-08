#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Define block size for each kernel launch
#define BLOCK_SIZE 256

// Kernel to compute the MSE for a sub-array (chunk) passed via pointers. 
// It computes the sum of squared differences for 'chunk_size' elements starting at the given pointers.
template <typename scalar_t>
__global__ void mse_chunk_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    const int64_t chunk_size,
    double* __restrict__ partial_result
) {
    __shared__ double shm[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = gridDim.x * blockDim.x;
    double local_sum = 0.0;

    // Grid-stride loop over the chunk with coalesced memory access
    #pragma unroll 4
    for (int i = idx; i < chunk_size; i += stride) {
        scalar_t pred = preds[i];  // Cache loads
        scalar_t tgt = tgts[i];
        double diff = static_cast<double>(pred) - static_cast<double>(tgt);
        local_sum += diff * diff;
    }
    
    shm[tid] = local_sum;
    __syncthreads();

    // Intra-block reduction using shared memory
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shm[tid] += shm[tid + offset];
        }
        __syncthreads();
    }

    // Atomically add the block's result to the partial result for this chunk
    if (tid == 0) {
        atomicAdd(partial_result, shm[0]);
    }
}

// Host function: partitions the input data into chunks and launches kernels on separate CUDA streams.
// Overlaps computation with asynchronous memory transfers of the partial results.
// 'nstreams' indicates the number of streams (and chunks) used for processing.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets, int nstreams = 4) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(), "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    // Determine chunk size per stream (ensuring full coverage of the data)
    int64_t chunk_size = (num_elements + nstreams - 1) / nstreams;

    // Allocate a device tensor to hold partial results (one double per stream)
    auto options = predictions.options().dtype(at::kDouble);
    torch::Tensor device_partial = torch::zeros({nstreams}, options);

    // Create CUDA streams
    std::vector<cudaStream_t> streams(nstreams);
    for (int i = 0; i < nstreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch a separate kernel for each chunk on its corresponding stream
    for (int i = 0; i < nstreams; i++) {
        int64_t offset = i * chunk_size;
        if (offset >= num_elements) break;
        int64_t current_chunk = std::min(chunk_size, num_elements - offset);
        int blocks = (current_chunk + BLOCK_SIZE - 1) / BLOCK_SIZE;

        AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_chunk_cuda", ([&] {
            mse_chunk_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, streams[i]>>>(
                predictions.data_ptr<scalar_t>() + offset,
                targets.data_ptr<scalar_t>() + offset,
                current_chunk,
                device_partial.data_ptr<double>() + i
            );
        }));
    }

    // Allocate pinned host memory for partial results to enable asynchronous copy
    auto host_partial = torch::empty({nstreams}, options.device(torch::kCPU)).pin_memory();

    // Asynchronously copy each partial result from device to host using its stream
    for (int i = 0; i < nstreams; i++) {
        cudaMemcpyAsync(
            host_partial.data_ptr<double>() + i,
            device_partial.data_ptr<double>() + i,
            sizeof(double),
            cudaMemcpyDeviceToHost,
            streams[i]
        );
    }

    // Synchronize all streams to ensure that kernels and memory copies are finished
    for (int i = 0; i < nstreams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Aggregate the partial results on the host
    double total_sum = 0.0;
    double* host_ptr = host_partial.data_ptr<double>();
    for (int i = 0; i < nstreams; i++) {
        total_sum += host_ptr[i];
    }

    // Compute final MSE value
    double mse = total_sum / static_cast<double>(num_elements);

    // Return the result as a tensor with the same type as the input predictions
    auto result = torch::tensor({mse}, predictions.options());
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward with stream overlap (CUDA)",
          pybind11::arg("predictions"), pybind11::arg("targets"), pybind11::arg("nstreams") = 4);
}
