#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that computes hinge loss per element and performs block-level reduction
// to sum the results. Only minimal __syncthreads() are used for shared memory consistency.
__global__ void hinge_loss_reduction_kernel(const float* predictions, const float* targets, float* global_sum, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Shared memory for block reduction; assume blockDim.x == 256
    __shared__ float shared_sum[256];

    // Each thread computes its hinge loss value if within range
    float hinge_val = 0.0f;
    if (idx < n) {
        hinge_val = fmaxf(0.0f, 1.0f - predictions[idx] * targets[idx]);
    }
    // Perform warp-level reduction using shuffle instructions
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        hinge_val += __shfl_down_sync(mask, hinge_val, offset);
    }

    // Write per-warp reduced sums to shared memory
    int warp_id = tid / warpSize;
    if ((tid & (warpSize - 1)) == 0) {
        shared_sum[warp_id] = hinge_val;
    }
    __syncthreads();

    // Final reduction: let the first warp load per-warp sums and combine them
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    if (tid < num_warps) {
        hinge_val = shared_sum[tid];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            hinge_val += __shfl_down_sync(mask, hinge_val, offset);
        }
        if (tid == 0) {
            atomicAdd(global_sum, hinge_val);
        }
    }
}

// Forward function that sets up kernel launch and computes the mean hinge loss
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    CHECK_INPUT(predictions);
    CHECK_INPUT(targets);

    int n = predictions.numel();

    // Allocate a tensor for the global sum (initialized to zero) on the same device
    auto options = predictions.options();
    torch::Tensor global_sum_tensor = torch::zeros({1}, options);
    float* global_sum_ptr = global_sum_tensor.data_ptr<float>();

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Launch the kernel
    hinge_loss_reduction_kernel<<<blocks, threads>>>(
        predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        global_sum_ptr,
        n
    );

    // Ensure kernel completion
    cudaDeviceSynchronize();

    // Retrieve the total hinge loss sum and compute the mean
    float total_loss = global_sum_tensor.item<float>();
    float mean_loss = total_loss / n;
    
    // Return the result as a scalar tensor
    return torch::full({}, mean_loss, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hinge Loss Forward Fast");
}
