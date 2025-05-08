#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define BLOCK_SIZE 256

// Warp-level reduction optimized for minimal branching
__device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level reduction with single synchronization
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    cg::thread_group warp = cg::tiled_partition<32>(cg::this_thread_block());
    float sum = warpReduceSum(val);

    if (warp.thread_rank() == 0)
        shared[warp.meta_group_rank()] = sum;

    cg::sync(cg::this_thread_block());

    if (warp.meta_group_rank() == 0) {
        sum = warp.thread_rank() < (blockDim.x + 31) / 32 ? shared[warp.thread_rank()] : 0.0f;
        sum = warpReduceSum(sum);
    }
    return sum;
}

// Vectorized sum accumulation with unrolled loops
__device__ float collescedSquareSum(const float* input, int numel) {
    constexpr int VEC_SIZE = 4;
    float sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process vectors
    const float4* input4 = reinterpret_cast<const float4*>(input);
    int vectorSteps = numel / (VEC_SIZE * stride);
    for (int i = 0; i < vectorSteps; ++i) {
        float4 vec = input4[idx + i * stride];
        sum += vec.x*vec.x + vec.y*vec.y + vec.z*vec.z + vec.w*vec.w;
    }
    idx += vectorSteps * VEC_SIZE * stride;

    // Process remaining elements
    while (idx < numel) {
        sum += input[idx] * input[idx];
        idx += stride;
    }
    return sum;
}

// Kernel functions
__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    float sum = collescedSquareSum(input, numel);
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0)
        atomicAdd(norm_out, sum);
}

__constant__ float d_norm;

__global__ void compute_sqrt_kernel(float* sum_ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_norm = sqrtf(*sum_ptr);
    }
}

__global__ void normalize_kernel(const float* input, float* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel)
        output[idx] = input[idx] * __frsqrt_rn(d_norm * d_norm);  // Fused multiply + rsqrt
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();

    int numel = input.numel();
    int threads = BLOCK_SIZE;
    int blocks = (numel + threads - 1) / threads;

    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);
    compute_sqrt_kernel<<<1, 1>>>(norm_ptr);
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Frobenius norm normalization");
}
