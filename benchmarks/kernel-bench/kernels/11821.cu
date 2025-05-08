#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to compute the KL divergence over a specific range [start, end) of elements
__global__ void kl_div_kernel_range(
    const float* log_predictions,
    const float* targets,
    float* partial,
    int start,
    int end) {

    int tid = threadIdx.x;
    int global_index = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    // Process the assigned chunk using grid-stride loop
    for (int i = start + global_index; i < end; i += stride) {
        float lp = log_predictions[i];
        float t = targets[i];
        local_sum += __expf(lp) - t * lp;
    }

    // Shared memory reduction for block-level sum
    __shared__ float shmem[256];
    shmem[tid] = local_sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset)
            shmem[tid] += shmem[tid + offset];
        __syncthreads();
    }

    // The first thread in each block atomically adds the block's sum to the partial result
    if (tid == 0) {
        atomicAdd(partial, shmem[0]);
    }
}

// Forward function that overlaps computation with asynchronous memory transfers using CUDA streams
torch::Tensor kl_div_cuda_forward_streamed(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    // Number of streams for pipelining
    const int num_streams = 4;
    int chunk_size = (n + num_streams - 1) / num_streams;

    // Create device tensor to hold partial results for each chunk
    auto partial_results = torch::zeros({num_streams}, log_predictions.options());

    // Create CUDA streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Allocate pinned host memory to asynchronously copy partial sums
    float* host_partial = nullptr;
    cudaHostAlloc((void**)&host_partial, num_streams * sizeof(float), cudaHostAllocDefault);

    const int threads = 256;

    // Launch one kernel per stream to process a chunk of the input
    for (int i = 0; i < num_streams; i++) {
        int start = i * chunk_size;
        if (start >= n) break;
        int end = start + chunk_size;
        if (end > n) end = n;
        int elements = end - start;
        int blocks = (elements + threads - 1) / threads;

        // Reset the partial result for this chunk
        cudaMemsetAsync(partial_results.data_ptr<float>() + i, 0, sizeof(float), streams[i]);

        // Launch the kernel for the assigned range on this stream
        kl_div_kernel_range<<<blocks, threads, 0, streams[i]>>>(
            log_predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            partial_results.data_ptr<float>() + i,
            start,
            end
        );

        // Asynchronously copy the computed partial sum from device to pinned host memory
        cudaMemcpyAsync(&host_partial[i],
                        partial_results.data_ptr<float>() + i,
                        sizeof(float),
                        cudaMemcpyDeviceToHost,
                        streams[i]);
    }

    // Synchronize all streams to ensure kernel execution and memory copies are complete
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Final reduction on the host
    float total_sum = 0.0f;
    for (int i = 0; i < num_streams; i++) {
        total_sum += host_partial[i];
    }
    
    cudaFreeHost(host_partial);

    // Create output tensor and store the averaged result
    auto output = torch::zeros({1}, log_predictions.options());
    float* out_ptr = output.data_ptr<float>();
    out_ptr[0] = total_sum / static_cast<float>(n);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_streamed, "Stream overlapped KL divergence forward (CUDA)");
}
