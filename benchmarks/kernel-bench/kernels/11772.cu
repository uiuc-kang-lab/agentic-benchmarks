#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define maximum number of elements that can be stored in constant memory
// For float, 16384 elements correspond to 64KB (assuming the device supports at least 64KB constant memory)

#define MAX_CONST_SIZE 16384

__constant__ float c_log_predictions[MAX_CONST_SIZE];
__constant__ float c_targets[MAX_CONST_SIZE];

constexpr int WARP_SIZE = 32;

// Kernel that computes the KL divergence using data loaded from constant memory
__global__ void const_mem_kl_kernel(float* output, const int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;

    // Each thread processes a grid-stride loop over the input, reading from constant memory
    for (int i = tid; i < n; i += total_threads) {
        float lp = c_log_predictions[i];
        float t  = c_targets[i];
        sum += expf(lp) - t * lp;
    }

    // Intra-warp reduction using shuffle operations
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Use shared memory to reduce across warps within the block
    extern __shared__ float sdata[];
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction by the first warp in the block
    if (threadIdx.x < blockDim.x / WARP_SIZE) {
        sum = sdata[threadIdx.x];
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Host function to launch the kernel. It copies the input tensors to constant memory and launches the kernel.
// This kernel assumes that the number of elements does not exceed MAX_CONST_SIZE.

torch::Tensor const_mem_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    TORCH_CHECK(log_predictions.is_contiguous(), "log_predictions must be contiguous");
    TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
    TORCH_CHECK(log_predictions.device().is_cuda(), "log_predictions must be a CUDA tensor");
    TORCH_CHECK(targets.device().is_cuda(), "targets must be a CUDA tensor");

    const int n = log_predictions.numel();
    TORCH_CHECK(n <= MAX_CONST_SIZE, "Input tensor size exceeds constant memory limit");

    // Copy input data to constant memory
    cudaMemcpyToSymbol(c_log_predictions, log_predictions.data_ptr<float>(), n * sizeof(float));
    cudaMemcpyToSymbol(c_targets, targets.data_ptr<float>(), n * sizeof(float));

    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    int num_warps = threads / WARP_SIZE;
    int shared_mem = num_warps * sizeof(float);

    const_mem_kl_kernel<<<blocks, threads, shared_mem>>>(output.data_ptr<float>(), n);

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &const_mem_kl_forward, "KL divergence using constant memory (CUDA)");
}
