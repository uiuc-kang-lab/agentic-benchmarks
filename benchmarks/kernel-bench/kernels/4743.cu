#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// CUDA kernel for computing sum of squares using warp shuffle
__global__ void compute_norm_kernel(const float* input, float* norm_out, int numel) {
    const unsigned int FULL_MASK = 0xffffffff;
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = tid % 32;
    
    float sum = 0.0f;
    
    // Compute partial sums
    while (idx < numel) {
        sum += input[idx] * input[idx];
        idx += blockDim.x * gridDim.x;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    
    // First thread in each warp writes to global memory
    if (lane_id == 0) {
        atomicAdd(norm_out, sum);
    }
}

// CUDA kernel for normalization using coalesced memory access
__global__ void normalize_kernel(const float* input, float* output, 
                               float norm, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numel) {
        output[idx] = input[idx] / norm;
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
    
    // Use multiple of warp size for thread count
    const int threads = 256;  // 8 warps per block
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