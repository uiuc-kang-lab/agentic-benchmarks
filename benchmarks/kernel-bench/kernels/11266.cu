#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// CUDA kernel using grid-stride loop and warp-level reduction
__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ predictions,
    const float* __restrict__ targets,
    float* output,
    int n_elements
) {
    // Each thread processes multiple elements using grid-stride loop
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float thread_sum = 0.0f;
    
    // Grid-stride loop for boundary-safe processing of large workloads
    for (int i = idx; i < n_elements; i += stride) {
        float diff = predictions[i] - targets[i];
        float abs_diff = fabsf(diff);
        thread_sum += (abs_diff < 1.0f) ? (0.5f * diff * diff) : (abs_diff - 0.5f);
    }
    
    // Warp-level reduction using shuffle
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // Shared memory reduction across warps within the block
    __shared__ float shared[32]; // Enough for up to 32 warps per block
    int lane = threadIdx.x & 31;         // thread index within the warp
    int warp_id = threadIdx.x >> 5;        // warp identifier within the block

    if (lane == 0) {
        shared[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // First warp aggregates results from each warp in the block
    float block_sum = 0.0f;
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        block_sum = (lane < num_warps) ? shared[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            // Atomically add the normalized block result to the global output
            atomicAdd(output, block_sum / n_elements);
        }
    }
}

// Host wrapper function
torch::Tensor smooth_l1_loss_cuda(
    torch::Tensor predictions,
    torch::Tensor targets
) {
    TORCH_CHECK(predictions.sizes() == targets.sizes(), "Input tensors must have the same shape");
    TORCH_CHECK(predictions.is_contiguous() && targets.is_contiguous(), "Input tensors must be contiguous");
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(), "Inputs must be CUDA tensors");

    int n = predictions.numel();
    auto output = torch::zeros({1}, predictions.options());

    const int block_size = 256;
    // Using grid-stride loop, we can launch a number of threads
    const int grid_size = (n + block_size - 1) / block_size;

    smooth_l1_loss_kernel<<<grid_size, block_size>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &smooth_l1_loss_cuda, "Smooth L1 Loss with stride loop (CUDA)");
}
