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
    shared_sum[tid] = hinge_val;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 32; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    // Unroll final warp (no __syncthreads() needed within a warp on modern CUDA architecture)
    if (tid < 32) {
        volatile float* vsmem = shared_sum;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }

    // Thread 0 of each block atomically adds the block's sum to the global sum
    if (tid == 0) {
        atomicAdd(global_sum, shared_sum[0]);
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
