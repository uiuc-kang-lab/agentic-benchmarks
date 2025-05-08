#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define maximum size for constant memory arrays (in number of floats).
// Adjust MAX_CONST_SIZE as needed to stay within hardware constant memory limits (typically 64KB).
#define MAX_CONST_SIZE 16384

// Constant memory declarations for frequently accessed, read-only data.
__constant__ float c_log_predictions[MAX_CONST_SIZE];
__constant__ float c_targets[MAX_CONST_SIZE];

// Kernel that uses constant memory for log_predictions and targets
__global__ void optimized_kl_div_kernel_constant(
    float* __restrict__ output,
    const int n) {

    // Calculate global thread index and stride (1D grid, 1D block)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float sum = 0.0f;
    
    // Iterate over elements with given stride using data from constant memory
    for (int i = idx; i < n; i += stride) {
        float lp = c_log_predictions[i];
        float tar = c_targets[i];
        sum += expf(lp) - tar * lp;
    }
    
    // Use shared memory for intra-block reduction
    extern __shared__ float sdata[];
    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Reduce partial sums in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Atomic addition of block's sum to global output
    if (threadIdx.x == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// Host function that copies data to constant memory and launches the kernel
torch::Tensor optimized_kl_div_cuda_forward_constant(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    // Ensure the data fits within the constant memory limit
    TORCH_CHECK(n <= MAX_CONST_SIZE, "Input size (", n, ") exceeds constant memory capacity (", MAX_CONST_SIZE, ")");
    
    // Create output tensor
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Copy input data to constant memory using cudaMemcpyToSymbol
    cudaMemcpyToSymbol(c_log_predictions, log_predictions.data_ptr<float>(), n * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_targets, targets.data_ptr<float>(), n * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Launch kernel with appropriate configuration
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    optimized_kl_div_kernel_constant<<<blocks, threads, shared_mem>>>(
        output.data_ptr<float>(),
        n
    );

    // Return the mean value by dividing the aggregated result by n
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_kl_div_cuda_forward_constant, "Optimized KL divergence forward using constant memory (CUDA)");
}
