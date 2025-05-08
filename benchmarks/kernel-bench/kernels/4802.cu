#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

template <unsigned int WF>
__device__ float warpReduceSum(float val) {
    for (unsigned int offset = WF/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum<32>(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
    
    if (wid == 0) {
        val = warpReduceSum<32>(val);
    }
    return val;
}

__global__ void compute_partial_sums(const float* input, float* partial_sums, int numel) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    while (idx < numel) {
        float val = input[idx];
        sum += val * val;
        idx += stride;
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sum;
    }
}

__global__ void final_reduce(const float* partial_sums, float* norm_out, int num_blocks) {
    float sum = 0.0f;
    int idx = threadIdx.x;
    
    while (idx < num_blocks) {
        sum += partial_sums[idx];
        idx += blockDim.x;
    }
    
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        *norm_out = sqrtf(sum);
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    int numel = input.numel();
    
    const int threads = BLOCK_SIZE;
    const int blocks = min(65535, (numel + threads - 1) / threads);
    auto partial_sums = torch::zeros({blocks}, input.options());

    // Compute partial sums per block
    compute_partial_sums<<<blocks, threads>>>(input.data_ptr<float>(),
                                            partial_sums.data_ptr<float>(),
                                            numel);

    // Final reduction to compute norm and normalize in single kernel
    final_reduce<<<1, threads>>>(partial_sums.data_ptr<float>(),
                               (float*)output.data_ptr<float>(),
                               blocks);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Frobenius norm with two-pass reduction");
}