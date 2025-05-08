#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Define maximum constant memory size (in number of floats).
// 16384 floats = 64KB, typical constant memory limit on many GPUs.
#define MAX_CONST_SIZE 16384

// Declare constant memory arrays for read-only input data.
__constant__ float c_log_predictions[MAX_CONST_SIZE];
__constant__ float c_targets[MAX_CONST_SIZE];

// Kernel using constant memory for frequently accessed read-only inputs.
__global__ void kldiv_const_mem_kernel(
    float* __restrict__ output,
    const int n) {

    const unsigned int warp_size = 32;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int warp_id = threadIdx.x / warp_size;
    const unsigned int warps_per_block = blockDim.x / warp_size;

    // Shared memory for intermediate warp results
    extern __shared__ float shared_warp_sums[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;

    // Process elements in a strided loop
    while (idx < n) {
        // Read from constant memory
        float log_pred = c_log_predictions[idx];
        float target = c_targets[idx];
        sum += expf(log_pred) - target * log_pred;
        idx += stride;
    }

    // Warp-level reduction using shuffle instructions
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread in each warp writes its result to shared memory
    if (lane_id == 0) {
        shared_warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction across warps (performed by the first warp)
    if (warp_id == 0) {
        sum = (lane_id < warps_per_block) ? shared_warp_sums[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Forward function that copies input tensors into constant memory and launches the kernel
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();

    // Ensure the input size fits in the constant memory allocation
    if(n > MAX_CONST_SIZE) {
        throw std::runtime_error("Input size exceeds constant memory capacity");
    }

    auto output = torch::zeros({1}, log_predictions.options());

    // Copy input data into constant memory. These are read-only throughout the kernel.
    cudaMemcpyToSymbol(c_log_predictions, log_predictions.data_ptr<float>(), n * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_targets, targets.data_ptr<float>(), n * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Launch configuration
    const int threads_per_block = 256; // Must be a multiple of warp size (32)
    const int blocks = std::min(256, (n + threads_per_block - 1) / threads_per_block);
    const int warps_per_block = threads_per_block / 32;
    const int shared_mem = warps_per_block * sizeof(float);

    kldiv_const_mem_kernel<<<blocks, threads_per_block, shared_mem>>>(output.data_ptr<float>(), n);

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA) with constant memory");
}
