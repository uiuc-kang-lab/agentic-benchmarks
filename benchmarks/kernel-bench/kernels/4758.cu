#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

__global__ void __launch_bounds__(128, 8) compute_norm_kernel(const float* input, float* norm_out, int numel) {
    const unsigned int FULL_MASK = 0xffffffff;
    __shared__ float warp_sums[4];  // For 128 threads = 4 warps per block
    
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Each thread accumulates its portion
    float sum = 0.0f;
    for (int idx = gid; idx < numel; idx += gridDim.x * blockDim.x) {
        float val = input[idx];
        sum += val * val;
    }
    
    // First level reduction - within warp using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    
    // Store the warp result
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction - only first warp reduces across all warps
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x / 32)) ? warp_sums[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(FULL_MASK, sum, offset);
        }
        
        if (lane_id == 0) {
            atomicAdd(norm_out, sum);
        }
    }
}

__global__ void normalize_kernel(const float* input, float* output, float norm, int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Vectorized loads and stores where possible
    for (; idx < numel; idx += stride) {
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