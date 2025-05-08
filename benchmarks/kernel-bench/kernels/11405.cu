#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Define maximum number of elements that can be stored in constant memory
// 16384 floats * 4 bytes = 65536 bytes, which is typically the hardware limit.
#define MAX_CHUNK_SIZE 16384

// Declare constant memory for read-only input arrays
__constant__ float d_const_log[MAX_CHUNK_SIZE];
__constant__ float d_const_target[MAX_CHUNK_SIZE];

// Kernel that reads from constant memory
__global__ void kl_div_kernel_const(int chunk_size, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    extern __shared__ float shared_sum[];
    float sum = 0.0f;

    // Process elements in a grid-stride loop
    for (int i = idx; i < chunk_size; i += stride) {
        float log_val = d_const_log[i];
        float target = d_const_target[i];
        sum += expf(log_val) - target * log_val;
    }

    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Atomic add the block result to global output
    if (threadIdx.x == 0) {
        atomicAdd(output, shared_sum[0]);
    }
}

// CUDA forward function
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int shared_mem = threads * sizeof(float);

    // Process input in one or more chunks depending on size
    if (n <= MAX_CHUNK_SIZE) {
        // Copy entire arrays to constant memory
        cudaMemcpyToSymbol(d_const_log, log_predictions.data_ptr<float>(), n * sizeof(float), 0, cudaMemcpyDeviceToDevice);
        cudaMemcpyToSymbol(d_const_target, targets.data_ptr<float>(), n * sizeof(float), 0, cudaMemcpyDeviceToDevice);

        int blocks = (n + threads - 1) / threads;
        kl_div_kernel_const<<<blocks, threads, shared_mem>>>(n, output.data_ptr<float>());
    } else {
        // Process inputs in chunks that fit in constant memory
        int chunks = (n + MAX_CHUNK_SIZE - 1) / MAX_CHUNK_SIZE;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            int offset = chunk * MAX_CHUNK_SIZE;
            int chunk_size = std::min(MAX_CHUNK_SIZE, n - offset);
            
            // Copy the current chunk to constant memory
            cudaMemcpyToSymbol(d_const_log, log_predictions.data_ptr<float>() + offset, chunk_size * sizeof(float), 0, cudaMemcpyDeviceToDevice);
            cudaMemcpyToSymbol(d_const_target, targets.data_ptr<float>() + offset, chunk_size * sizeof(float), 0, cudaMemcpyDeviceToDevice);
            
            int blocks = (chunk_size + threads - 1) / threads;
            kl_div_kernel_const<<<blocks, threads, shared_mem>>>(chunk_size, output.data_ptr<float>());
        }
    }

    // Normalize the sum by the total number of elements
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA) using constant memory");
}
