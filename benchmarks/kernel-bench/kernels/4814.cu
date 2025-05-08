#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256

// Enhanced warp reduction with compiler optimization
__device__ float warpReduce(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__device__ float blockReduce(float val) {
    __shared__ float shared[32];
    int wid = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    val = warpReduce(val);
    __syncthreads();

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (blockDim.x / 32)) ? shared[lane] : 0.0f;
    val = warpReduce(val);
    return val;
}

__device__ float computeSum(const float* input, int numel) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process 4 elements at a time with vectorized loads
    float4* input4 = (float4*)input;
    int numel4 = numel / 4;
    for (int i = idx; i < numel4; i += stride) {
        float4 val = input4[i];
        sum += val.x*val.x + val.y*val.y + val.z*val.z + val.w*val.w;
    }
    // Handle remaining elements
    int base = numel4 * 4;
    for (int i = idx + base; i < numel; i += stride) {
        float val = input[i];
        sum += val * val;
    }
    return sum;
}

__global__ void compute_norm_kernel(const float* input, float* partial_sums, int numel) {
    float sum = computeSum(input, numel);
    sum = blockReduce(sum);
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = sum;
    }
}

__global__ void sqrt_kernel(float* val) {
    *val = sqrtf(*val);
}

__constant__ float d_norm;

__global__ void normalize_kernel(const float* input, float* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx] / d_norm;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);
    auto norm = torch::zeros(1, input.options());
    
    const int numel = input.numel();
    const int threads = BLOCK_SIZE;
    const int blocks = (numel + threads - 1) / threads;

    compute_norm_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                           norm.data_ptr<float>(),
                                           numel);
    
    sqrt_kernel<<<1, 1>>>(norm.data_ptr<float>());
    
    cudaMemcpyToSymbol(d_norm, norm.data_ptr<float>(), sizeof(float), 0, cudaMemcpyDeviceToDevice);
    
    normalize_kernel<<<blocks, threads>>>(input.data_ptr<float>(),
                                        output.data_ptr<float>(),
                                        numel);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Frobenius normalization");
}