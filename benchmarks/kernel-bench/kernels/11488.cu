#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Kernel to compute KL divergence over a chunk
// Processes 'count' elements from the beginning of the provided arrays (which represent a chunk).
__global__ void kl_div_kernel_chunk(const float* log_preds, const float* targets, float* partial, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sdata[];
    float sum = 0.0f;
    // Grid-stride loop over the chunk
    for (int i = tid; i < count; i += blockDim.x * gridDim.x) {
        float log_pred = log_preds[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        __syncthreads();
    }
    
    // The first thread of the block atomically adds its block's sum to the global partial sum
    if (threadIdx.x == 0) {
        atomicAdd(partial, sdata[0]);
    }
}

// Forward function that overlaps memory transfers with kernel computation using CUDA streams.
// This version assumes the input tensors (log_predictions and targets) reside in CPU pinned memory, so that
// asynchronous memory copies from host to device can be overlapped with kernel execution.

torch::Tensor kl_div_cuda_forward(torch::Tensor log_predictions, torch::Tensor targets) {
    // Total number of elements in the input
    const int n = log_predictions.numel();
    
    // Number of streams (and chunks) chosen for overlapping transfers and computation
    const int num_streams = 4;
    int chunk_size = (n + num_streams - 1) / num_streams;

    // Allocate pinned host memory for partial results from each stream
    float* h_partial = nullptr;
    cudaHostAlloc((void**)&h_partial, num_streams * sizeof(float), cudaHostAllocDefault);
    for (int i = 0; i < num_streams; i++) {
        h_partial[i] = 0.0f;
    }

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Vectors to store device pointers for each stream for cleanup
    std::vector<float*> d_log_preds_vec(num_streams, nullptr);
    std::vector<float*> d_targets_vec(num_streams, nullptr);
    std::vector<float*> d_partial_vec(num_streams, nullptr);

    // Assume the inputs are contiguous and reside in host pinned memory
    const float* h_log_preds = log_predictions.data_ptr<float>();
    const float* h_targets = targets.data_ptr<float>();

    // Launch kernels concurrently on multiple streams for each chunk
    for (int i = 0; i < num_streams; i++) {
        int offset = i * chunk_size;
        int current_chunk = (offset + chunk_size > n) ? (n - offset) : chunk_size;
        if (current_chunk <= 0)
            continue;
        
        float *d_log_preds, *d_targets, *d_partial;
        cudaMalloc((void**)&d_log_preds, current_chunk * sizeof(float));
        cudaMalloc((void**)&d_targets, current_chunk * sizeof(float));
        cudaMalloc((void**)&d_partial, sizeof(float));
        
        d_log_preds_vec[i] = d_log_preds;
        d_targets_vec[i] = d_targets;
        d_partial_vec[i] = d_partial;
        
        // Initialize the device partial result to 0
        cudaMemsetAsync(d_partial, 0, sizeof(float), streams[i]);
        
        // Asynchronously copy the current chunk of data from host to device
        cudaMemcpyAsync(d_log_preds, h_log_preds + offset, current_chunk * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_targets, h_targets + offset, current_chunk * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        
        // Launch the kernel for the current chunk
        int threads = 256; // Align to warp size
        int blocks = (current_chunk + threads - 1) / threads;
        int shared_mem = threads * sizeof(float);
        kl_div_kernel_chunk<<<blocks, threads, shared_mem, streams[i]>>>(d_log_preds, d_targets, d_partial, current_chunk);
        
        // Asynchronously copy the partial result back to host pinned memory
        cudaMemcpyAsync(h_partial + i, d_partial, sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize all streams to ensure that all transfers and kernel executions have completed
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Final reduction on the host: sum up all partial results
    float total = 0.0f;
    for (int i = 0; i < num_streams; i++) {
        total += h_partial[i];
    }

    // Cleanup allocated device memory
    for (int i = 0; i < num_streams; i++) {
        if(d_log_preds_vec[i]) cudaFree(d_log_preds_vec[i]);
        if(d_targets_vec[i]) cudaFree(d_targets_vec[i]);
        if(d_partial_vec[i]) cudaFree(d_partial_vec[i]);
    }
    cudaFreeHost(h_partial);

    // Create and return the output tensor with the final result (normalized by n)
    auto output = torch::full({1}, total / static_cast<float>(n), log_predictions.options());
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward with stream overlap (CUDA)");
}
