#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for computing sum of squares with unrolled reduction
__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    __shared__ float shared_sum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    shared_sum[tid] = 0.0f;
    
    // Compute partial sums with pragma unroll
    #pragma unroll 4
    while (idx < numel) {
        shared_sum[tid] += input[idx] * input[idx];
        idx += blockDim.x * gridDim.x;
    }
    __syncthreads();
    
    // Manually unrolled reduction
    if (tid < 128) { shared_sum[tid] += shared_sum[tid + 128]; } __syncthreads();
    if (tid < 64) { shared_sum[tid] += shared_sum[tid + 64]; } __syncthreads();
    if (tid < 32) {
        // Warp-level operations don't need synchronization
        shared_sum[tid] += shared_sum[tid + 32];
        shared_sum[tid] += shared_sum[tid + 16];
        shared_sum[tid] += shared_sum[tid + 8];
        shared_sum[tid] += shared_sum[tid + 4];
        shared_sum[tid] += shared_sum[tid + 2];
        shared_sum[tid] += shared_sum[tid + 1];
    }
    
    // Write result for this block
    if (tid == 0) {
        atomicAdd(norm_out, shared_sum[0]);
    }
}

// CUDA kernel for normalization with unrolled processing
__global__ void normalize_kernel(const float* input, float* output, 
                               float norm, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    #pragma unroll 4
    while (idx < numel) {
        output[idx] = input[idx] / norm;
        idx += blockDim.x * gridDim.x;
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
    const int blocks = min(65535, (numel + threads - 1) / threads);

    compute_norm_kernel<<<blocks, threads>>>(input_ptr, norm_ptr, numel);
    
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);
    
    normalize_kernel<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Frobenius norm normalization");
}