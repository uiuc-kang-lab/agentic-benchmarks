#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the maximum number of elements that can be stored in constant memory
// For float type, 16384 elements = 64KB (16384 * 4 bytes)
#define MAX_CONST_SIZE 16384

// Declare constant memory for read-only data
__constant__ float d_log_predictions_const[MAX_CONST_SIZE];
__constant__ float d_targets_const[MAX_CONST_SIZE];

// CUDA kernel that uses constant memory for frequently accessed, read-only data
__global__ void kl_div_kernel_const(
    float* __restrict__ output,
    const int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float shared_sum[]; // Shared memory for reduction

    float sum = 0.0f;
    
    // Grid-stride loop over elements using constant memory accesses
    for (int i = idx; i < n; i += blockDim.x * gridDim.x) {
        // Read from constant memory
        float log_pred = d_log_predictions_const[i];
        float target = d_targets_const[i];
        sum += expf(log_pred) - target * log_pred;
    }

    // Store the partial sum in shared memory
    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    // Perform parallel reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Atomic addition of block result to global output
    if (threadIdx.x == 0) {
        atomicAdd(output, shared_sum[0]);
    }
}

// Host function to launch the kernel
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    // Ensure the input size fits within the constant memory limit
    TORCH_CHECK(n <= MAX_CONST_SIZE, "Input size exceeds constant memory capacity (" 
                "MAX_CONST_SIZE = " #MAX_CONST_SIZE ")");

    auto output = torch::zeros({1}, log_predictions.options());

    // Copy the input tensors from device memory to constant memory
    cudaMemcpyToSymbol(d_log_predictions_const, log_predictions.data_ptr<float>(),
                         n * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(d_targets_const, targets.data_ptr<float>(),
                         n * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);

    // Launch the constant-memory optimized kernel
    kl_div_kernel_const<<<blocks, threads, shared_mem>>>(output.data_ptr<float>(), n);

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA with constant memory)");
}
