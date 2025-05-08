#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized CUDA kernel for computing sum of squares with 1D block and thread mapping
__global__ void compute_norm_kernel_optimized(const float* input, float* norm_out, int numel) {
    extern __shared__ float shared_sum[];
    
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    shared_sum[tid] = 0.0f;

    // Loop through elements per thread
    if (idx < numel) {
        shared_sum[tid] = input[idx] * input[idx];
    }
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        atomicAdd(norm_out, shared_sum[0]);
    }
}

// CUDA kernel for normalization with 1D block and thread mapping
__global__ void normalize_kernel_optimized(const float* input, float* output, float norm, int numel) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < numel) {
        output[idx] = input[idx] / norm;
    }
}

torch::Tensor forward_optimized(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    // Create output tensor with same shape as input
    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());
    
    // Get raw pointers
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    float* norm_ptr = norm_tensor.data_ptr<float>();
    
    // Calculate total number of elements
    int numel = input.numel();
    
    // Calculate grid and block dimensions
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    // First kernel: compute sum of squares
    compute_norm_kernel_optimized<<<blocks, threads, threads * sizeof(float)>>>(input_ptr, norm_ptr, numel);
    
    // Get norm value and compute square root
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);
    
    // Second kernel: normalize the tensor
    normalize_kernel_optimized<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_optimized", &forward_optimized, "Optimized Frobenius norm normalization");
}