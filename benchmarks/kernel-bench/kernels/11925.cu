#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <cmath>

// Kernel that computes Triplet Margin Loss for a chunk of the batch
// batch_offset: starting index in the batch
// chunk_size: number of samples to process in this kernel launch
// Each warp (32 threads) computes one sample's loss

template <typename scalar_t>
__global__ void triplet_margin_loss_kernel_overlap(
    const scalar_t* __restrict__ anchor,
    const scalar_t* __restrict__ positive,
    const scalar_t* __restrict__ negative,
    scalar_t* __restrict__ output,
    const float margin,
    const int batch_offset,
    const int chunk_size,
    const int feat_size) {

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / 32; // each warp processes one sample
    int lane_id = global_thread_id % 32;

    if (warp_id >= chunk_size) return;

    int sample_idx = batch_offset + warp_id;  // absolute sample index
    scalar_t dist_pos = 0;
    scalar_t dist_neg = 0;
    
    // Each thread in the warp processes a subset of features
    for (int i = lane_id; i < feat_size; i += 32) {
        int idx = sample_idx * feat_size + i;
        scalar_t a = anchor[idx];
        scalar_t p = positive[idx];
        scalar_t n = negative[idx];
        scalar_t d_pos = a - p;
        scalar_t d_neg = a - n;
        dist_pos += d_pos * d_pos;
        dist_neg += d_neg * d_neg;
    }
    
    // Warp-level reduction using shuffles
    for (int offset = 16; offset > 0; offset /= 2) {
        dist_pos += __shfl_down_sync(0xffffffff, dist_pos, offset);
        dist_neg += __shfl_down_sync(0xffffffff, dist_neg, offset);
    }
    
    if (lane_id == 0) {
        scalar_t loss = sqrt(dist_pos) - sqrt(dist_neg) + margin;
        loss = loss < 0 ? 0 : loss;
        output[sample_idx] = loss;
    }
}

// Forward function implementing overlapping of computation and memory transfers.
// The batch is split into chunks, and for each chunk:
//  1. The kernel is launched asynchronously on a CUDA stream
//  2. The computed chunk is asynchronously copied from device to pinned host memory
// Finally, the host computes the mean of the loss values.

torch::Tensor triplet_margin_loss_cuda(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    float margin) {

    TORCH_CHECK(anchor.is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.is_cuda(), "negative must be a CUDA tensor");

    int batch_size = anchor.size(0);
    int feat_size = anchor.size(1);

    // Allocate device output tensor
    auto output_device = torch::empty({batch_size}, anchor.options());
    // Allocate pinned host memory to overlap memory transfers
    auto output_host = torch::empty({batch_size}, torch::TensorOptions()
                                      .dtype(anchor.dtype())
                                      .device(torch::kCPU)
                                      .pinned_memory(true));

    // Set chunk parameters
    int chunk_size = 64;
    if (chunk_size > batch_size) chunk_size = batch_size;
    int num_chunks = (batch_size + chunk_size - 1) / chunk_size;

    // Create two CUDA streams to pipeline kernel execution and memory copies
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    int threads_per_block = 128;
    
    // Process each chunk independently
    for (int i = 0; i < num_chunks; i++) {
        int batch_offset = i * chunk_size;
        int current_chunk = std::min(chunk_size, batch_size - batch_offset);
        int total_threads = current_chunk * 32;  // one warp (32 threads) per sample
        int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

        // Launch kernel asynchronously on one of the streams
        AT_DISPATCH_FLOATING_TYPES(anchor.scalar_type(), "triplet_margin_loss_kernel_overlap", ([&] {
            triplet_margin_loss_kernel_overlap<scalar_t><<<blocks, threads_per_block, 0, streams[i % 2]>>>(
                anchor.data_ptr<scalar_t>(),
                positive.data_ptr<scalar_t>(),
                negative.data_ptr<scalar_t>(),
                output_device.data_ptr<scalar_t>(),
                margin,
                batch_offset,
                current_chunk,
                feat_size);
        }));

        // Asynchronously copy the computed chunk from device to pinned host memory
        size_t elem_size = anchor.element_size();
        size_t chunk_bytes = current_chunk * elem_size;
        cudaMemcpyAsync(
            static_cast<char*>(output_host.data_ptr()) + batch_offset * elem_size,
            static_cast<char*>(output_device.data_ptr()) + batch_offset * elem_size,
            chunk_bytes,
            cudaMemcpyDeviceToHost,
            streams[i % 2]
        );
    }

    // Ensure all streams have completed their work
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);

    // Destroy the CUDA streams
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);

    // Compute mean loss on host
    if (anchor.scalar_type() == at::kFloat) {
        float sum = 0.0f;
        float* host_ptr = output_host.data_ptr<float>();
        for (int i = 0; i < batch_size; i++) {
            sum += host_ptr[i];
        }
        float mean = sum / batch_size;
        return torch::tensor(mean, torch::TensorOptions().dtype(anchor.dtype()).device(torch::kCPU));
    } else {
        double sum = 0.0;
        double* host_ptr = output_host.data_ptr<double>();
        for (int i = 0; i < batch_size; i++) {
            sum += host_ptr[i];
        }
        double mean = sum / batch_size;
        return torch::tensor(mean, torch::TensorOptions().dtype(anchor.dtype()).device(torch::kCPU));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &triplet_margin_loss_cuda, "Triplet margin loss forward (CUDA) with stream overlap");
}
