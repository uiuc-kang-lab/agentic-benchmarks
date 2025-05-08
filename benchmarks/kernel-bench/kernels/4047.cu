#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Warp-level reduction using __shfl_down_sync
__inline__ __device__ float warpReduceSum(float val) {
    // All threads in a warp participate, mask 0xffffffff
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Fused kernel: computes ELU activation elementwise and simultaneously reduces the output
// using shared memory for intra-block reduction and warp-level primitives for final reduction
__global__ void elu_kernel_fused(const float* __restrict__ x,
                                  float* __restrict__ out,
                                  float alpha,
                                  int n,
                                  float* global_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    // Process multiple elements per thread
    for (int i = idx; i < n; i += stride) {
        float val = x[i];
        float activated = (val > 0.0f) ? val : alpha * (expf(val) - 1.0f);
        out[i] = activated;
        local_sum += activated;
    }

    // Warp-level reduction within each warp
    local_sum = warpReduceSum(local_sum);

    // Intra-block reduction using shared memory
    __shared__ float shared[32]; // Assuming maximum 32 warps per block
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warpId] = local_sum;
    }
    __syncthreads();

    // Let the first warp finalize the block reduction
    float block_sum = 0.0f;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < numWarps) {
        block_sum = shared[lane];
        block_sum = warpReduceSum(block_sum);
    }

    // First thread of block adds the block's sum to the global sum
    if (threadIdx.x == 0) {
        atomicAdd(global_sum, block_sum);
    }
}

// Host function that launches the fused kernel
// Returns a tuple: {ELU-activated tensor, sum of all activated values}
std::tuple<torch::Tensor, torch::Tensor> elu_cuda_fused(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    auto sum_tensor = torch::zeros({1}, x.options());
    int n = x.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    elu_kernel_fused<<<blocks, threads>>>(x.data_ptr<float>(),
                                          out.data_ptr<float>(),
                                          alpha,
                                          n,
                                          sum_tensor.data_ptr<float>());

    return std::make_tuple(out, sum_tensor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_fused, "ELU activation fused with reduction (CUDA)");
}
