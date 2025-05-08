#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Fused kernel: computes swish activation and performs a block-level reduction
// to accumulate the sum of all swish outputs using shared memory and warp-level primitives.
__global__ void swish_fused_kernel(const float* __restrict__ x, float* __restrict__ y, float* __restrict__ global_sum, int64_t n) {
    // Assuming blockDim.x is 256
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    // Process multiple elements per thread
    for (int i = idx; i < n; i += stride) {
        const float val = x[i];
        // Cache the expensive exponential computation
        const float exp_val = expf(-val);
        // Cache the sigmoid computation
        const float sigmoid = 1.0f / (1.0f + exp_val);
        // Compute final result using cached values
        const float res = val * sigmoid;
        y[i] = res;
        local_sum += res;
    }

    // Each thread writes its partial sum to shared memory
    sdata[tid] = local_sum;
    __syncthreads();

    // Intra-block reduction using shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction for the final 32 elements using __shfl_down_sync
    if (tid < 32) {
        float sum_val = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
        }
        if (tid == 0) {
            // Atomically add the block's sum to the global accumulator
            atomicAdd(global_sum, sum_val);
        }
    }
}

// Swish activation forward pass that fuses element-wise computation with a reduction
// of the activated values. The primary output (y) is the same as the original 25_Swish result.
// An additional global sum of the outputs is computed for potential further use.

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    // Global reduction sum (if needed by later stages) is initialized to zero
    auto global_sum = torch::zeros({1}, x.options());
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    swish_fused_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), global_sum.data_ptr<float>(), n);

    // The kernel returns y, which holds the correct swish activation results.
    // The computed global_sum can be used for additional post-processing if required.
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Fused swish activation with reduction (CUDA)");
}
