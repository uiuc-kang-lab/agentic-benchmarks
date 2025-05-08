#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ __forceinline__ float warp_gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = x + coeff * x_cubed;
    inner *= sqrt_2_over_pi;
    float tanh_val = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}

__global__ void gelu_warp_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                const int n) {
    __shared__ float warp_data[32][32];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = threadIdx.x / 32;
    const unsigned int lane = tid % 32;
    const unsigned int warps_per_block = blockDim.x / 32;
    const unsigned int global_warp_id = blockIdx.x * warps_per_block + wid;
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int idx = (global_warp_id * 32 * 4) + (i * 32) + lane;
        if (idx < n) {
            float val = input[idx];
            warp_data[wid][lane] = val;
            __syncwarp();
            
            float result = warp_gelu(val);
            
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                float temp = __shfl_down_sync(0xffffffff, result, offset);
                if (lane < offset) {
                    result = fmaxf(result, temp);
                }
            }
            
            output[idx] = result;
        }
    }
}

torch::Tensor gelu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    
    auto output = torch::empty_like(input);
    const int n = input.numel();
    
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int num_elements_per_block = warps_per_block * 32 * 4;
    const int blocks = (n + num_elements_per_block - 1) / num_elements_per_block;
    
    gelu_warp_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Warp-optimized GELU forward CUDA implementation");
}