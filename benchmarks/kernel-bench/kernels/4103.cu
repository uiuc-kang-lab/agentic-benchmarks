#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel computes the ELU activation elementwise and also accumulates a reduction (the sum of all activated values)
// It uses a grid-stride loop for processing, shared memory for intra-block reductions,
// and warp-level primitives (__shfl_down_sync) for efficient reduction within a warp.

__global__ void elu_activation_reduction_kernel(const float* __restrict__ x,
                                                  float* __restrict__ out,
                                                  float* __restrict__ global_sum,
                                                  float alpha,
                                                  int n) {
    float local_sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process multiple elements per thread with grid-stride loop
    for (int i = idx; i < n; i += stride) {
        float val = x[i];
        float activated = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
        out[i] = activated;
        local_sum += activated;
    }

    // Perform warp-level reduction on local_sum
    unsigned int mask = 0xffffffff;
    int lane = threadIdx.x & 31; // lane index in the warp
    // Reduce within warp using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Each warp's lane 0 writes its sum to shared memory
    __shared__ float shared[32];  // enough for up to 32 warps per block
    int warpId = threadIdx.x / 32;
    if (lane == 0) {
        shared[warpId] = local_sum;
    }
    __syncthreads();

    // First warp of the block reduces the partial sums from each warp
    if (threadIdx.x < 32) {
        int numWarps = (blockDim.x + 31) / 32;
        float block_sum = (threadIdx.x < numWarps) ? shared[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(global_sum, block_sum);
        }
    }
}

// The wrapper function allocates the output tensor and a scalar tensor to hold the reduction sum.
// It launches the kernel and returns both the ELU-activated tensor and the sum of its elements.

std::vector<torch::Tensor> elu_activation_reduction_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    auto reduction = torch::zeros({1}, x.options());

    int n = x.numel();
    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;

    // Launch kernel: shared memory is statically allocated, so no dynamic shared mem size needed
    elu_activation_reduction_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        reduction.data_ptr<float>(),
        alpha,
        n
    );

    return {out, reduction};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_activation_reduction_cuda, "ELU activation with reduction optimization (CUDA)");
}
