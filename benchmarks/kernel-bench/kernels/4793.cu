#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>

__constant__ float d_norm;

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    __shared__ float block_sum;
    
    float thread_sum = 0.0f;
    const int tid = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    
    // Process 4 elements per thread with stride
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < numel)
            thread_sum += input[idx] * input[idx];
    }
    
    // Warp-level reduction
    float warp_sum = warpReduceSum(thread_sum);
    
    // First thread in warp writes to shared memory
    if (threadIdx.x % 32 == 0)
        atomicAdd(&block_sum, warp_sum);
    
    __syncthreads();
    
    // Single atomicAdd per block
    if (threadIdx.x == 0) {
        atomicAdd(norm_out, block_sum);
        block_sum = 0;  // Reset for next kernel call
    }
}

__global__ void normalize_kernel(const float* input, float* output, int numel) {
    const int tid = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = tid + i * blockDim.x;
        if (idx < numel)
            output[idx] = input[idx] / d_norm;
    }
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
    const int threads = 256;
    const int blocks = (numel + threads * 4 - 1) / (threads * 4);

    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);

    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrtf(norm_val);
    cudaMemcpyToSymbol(d_norm, &norm_val, sizeof(float));

    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Frobenius norm with grid-stride and warp reduction");
}