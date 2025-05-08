#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel that processes a chunk of the input starting at 'offset' and spanning 'num_elements'.
// It reduces the KL divergence for that chunk and accumulates the result into partial_result using atomicAdd.
__global__ void kl_div_kernel_with_offset(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ partial_result,
    const int offset,
    const int num_elements) {

    extern __shared__ float shared_sum[];
    float sum = 0.0f;

    // Each thread processes 4 elements in a grid-stride loop style
    int tid = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < num_elements) {
            int global_idx = offset + idx;
            float log_pred = log_predictions[global_idx];
            float target = targets[global_idx];
            sum += expf(log_pred) - target * log_pred;
        }
    }

    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    // Intra-block reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write block's reduced sum to partial_result with atomicAdd
    if (threadIdx.x == 0) {
        atomicAdd(partial_result, shared_sum[0]);
    }
}

// Forward function that overlaps kernel computation with asynchronous memory transfers using CUDA streams
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    // Total number of elements
    const int n = log_predictions.numel();
    
    // Use 2 streams for pipelining
    const int nStreams = 2;
    int chunk_size = (n + nStreams - 1) / nStreams;

    // Allocate an intermediate device tensor to hold partial results for each stream (one float per stream)
    auto partial_tensor = torch::zeros({nStreams}, log_predictions.options());

    // Create CUDA streams
    std::vector<cudaStream_t> streams(nStreams);
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Allocate pinned host memory to overlap memory transfer with computation
    float* host_partial = nullptr;
    cudaMallocHost((void**)&host_partial, nStreams * sizeof(float));

    const int threads = 256;
    // Launch a kernel for each stream to process a chunk of the input
    for (int i = 0; i < nStreams; i++) {
        int offset = i * chunk_size;
        if (offset >= n) break;
        int num_elements = std::min(chunk_size, n - offset);
        int blocks = (num_elements + threads * 4 - 1) / (threads * 4);
        
        // Reset the partial result for this stream asynchronously
        cudaMemsetAsync(partial_tensor.data_ptr<float>() + i, 0, sizeof(float), streams[i]);

        // Launch the kernel in the corresponding stream
        kl_div_kernel_with_offset<<<blocks, threads, threads * sizeof(float), streams[i]>>>(
            log_predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            partial_tensor.data_ptr<float>() + i,
            offset,
            num_elements
        );

        // Asynchronously copy the partial result from device to host pinned memory
        cudaMemcpyAsync(&host_partial[i], partial_tensor.data_ptr<float>() + i, sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize all streams to ensure that kernel execution and memory transfers are complete
    for (int i = 0; i < nStreams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Final reduction on host: sum the partial results
    float total = 0.0f;
    for (int i = 0; i < nStreams; i++) {
        total += host_partial[i];
    }

    cudaFreeHost(host_partial);

    // Return result as a tensor (average KL divergence over n elements)
    auto result = torch::full({1}, total / static_cast<float>(n), log_predictions.options());
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward with CUDA streams");
}
