#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 512  // Increased from 256

// Warp-level reduction using shuffle intrinsics
template <unsigned int WF>
__device__ float warpReduceSum(float val) {
    for (unsigned int offset = WF/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    
    val = warpReduceSum<32>(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0;
    if (threadIdx.x < 32) val = warpReduceSum<32>(val);
    return val;
}

__device__ float computeBlockSum(const float* input, int numel) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        float val = __ldg(&input[idx]);
        sum += val * val;
    }
    return sum;
}

__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    float block_sum = computeBlockSum(input, numel);
    block_sum = blockReduceSum(block_sum);
    if (threadIdx.x == 0) atomicAdd(norm_out, block_sum);
}

__constant__ float d_norm;

__global__ void normalize_kernel(const float* input, float* output, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) output[idx] = input[idx] / d_norm;
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());
    
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();

    int numel = input.numel();
    const int threads = BLOCK_SIZE;
    const int blocks = min(65535, (numel + threads - 1) / threads);

    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);
    
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrtf(norm_val);
    
    cudaMemcpyToSymbol(d_norm, &norm_val, sizeof(float));
    
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, numel);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm with optimized block size");
}