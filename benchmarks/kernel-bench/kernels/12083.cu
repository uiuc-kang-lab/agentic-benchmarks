#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel to compute the partial sum of the hinge loss for a chunk of data
__global__ void hinge_loss_reduction_kernel(const float* predictions, const float* targets, int start, int count, float* partial_sum) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    // Grid-stride loop over the chunk
    for (int idx = i; idx < count; idx += blockDim.x * gridDim.x) {
        int global_idx = start + idx;
        float prod = predictions[global_idx] * targets[global_idx];
        float loss = 1.0f - prod;
        if (loss < 0.0f) loss = 0.0f;
        sum += loss;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Each block contributes its result with one atomicAdd per block
        atomicAdd(partial_sum, sdata[0]);
    }
}

// The forward function partitions the input into chunks and uses CUDA streams to overlap
// kernel execution with asynchronous memory transfers of the partial sums.
// The final hinge loss mean is computed on the host, ensuring correct result and precision.

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();

    // Set a chunk size (e.g., 1M elements per chunk) for pipelining workload
    int chunk_size = 1 << 20;
    if (n < chunk_size) {
        chunk_size = n;
    }
    int num_chunks = (n + chunk_size - 1) / chunk_size;

    // Allocate a device tensor to store partial sums for each chunk and initialize to 0
    auto partial_sums = torch::zeros({num_chunks}, predictions.options());
    float* d_partial_sums = partial_sums.data_ptr<float>();

    // Allocate pinned host memory for asynchronous transfer of partial sums
    float* h_partial_sums;
    cudaMallocHost((void**)&h_partial_sums, num_chunks * sizeof(float));

    // Get raw pointers for the input tensors
    const float* d_predictions = predictions.data_ptr<float>();
    const float* d_targets = targets.data_ptr<float>();

    // Create a CUDA stream for each chunk to overlap computation and memory transfers
    std::vector<cudaStream_t> streams(num_chunks);
    for (int i = 0; i < num_chunks; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Launch kernels for each chunk on its dedicated stream
    for (int i = 0; i < num_chunks; i++) {
        int start = i * chunk_size;
        int count = (start + chunk_size > n) ? (n - start) : chunk_size;
        
        int threads = 256;
        int blocks = (count + threads - 1) / threads;
        
        // Launch the reduction kernel: dynamic shared memory = threads * sizeof(float)
        hinge_loss_reduction_kernel<<<blocks, threads, threads * sizeof(float), streams[i]>>>(
            d_predictions, d_targets, start, count, d_partial_sums + i
        );
        
        // Asynchronously copy the partial sum from device to pinned host memory
        cudaMemcpyAsync(&h_partial_sums[i], d_partial_sums + i, sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Synchronize all streams and destroy them
    for (int i = 0; i < num_chunks; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Accumulate partial sums into a total sum on host
    float total_sum = 0.0f;
    for (int i = 0; i < num_chunks; i++) {
        total_sum += h_partial_sums[i];
    }

    // Free the pinned host memory
    cudaFreeHost(h_partial_sums);

    // Compute the mean hinge loss
    float mean = total_sum / static_cast<float>(n);
    
    // Return the mean as a tensor
    return torch::tensor({mean}, predictions.options());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Overlapped Hinge Loss Forward");
}
