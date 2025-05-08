#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Optimized reduction kernel using warp shuffle operations
__global__ void compute_norm_kernel_optimized(const float* input, float* norm_out, int numel) {
    __shared__ float sdata[256];
    const unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (blockDim.x * 2) + tid;  // Process 2 elements per thread
    float sum = 0.0f;

    // Each thread processes two elements to reduce memory transactions
    while (idx < numel) {
        float val1 = input[idx];
        float val2 = (idx + blockDim.x < numel) ? input[idx + blockDim.x] : 0.0f;
        sum += val1 * val1 + val2 * val2;
        idx += blockDim.x * gridDim.x * 2;
    }

    sdata[tid] = sum;
    __syncthreads();

    // Optimized reduction using unrolling
    #pragma unroll
    for (int offset = blockDim.x/2; offset >= 64; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    // Warp-level reduction using shuffle operations
    if (tid < 32) {
        float val = sdata[tid];
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) {
            atomicAdd(norm_out, val);
        }
    }
}

// Optimized normalization kernel using vectorized loads/stores
__global__ void normalize_kernel_optimized(const float* input, float* output, float norm, int numel) {
    const unsigned int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    if (idx + blockDim.x < numel) {
        // Vector loads and stores for better memory bandwidth utilization
        float2 in_vec = *reinterpret_cast<const float2*>(&input[idx]);
        float2 out_vec;
        out_vec.x = in_vec.x / norm;
        out_vec.y = in_vec.y / norm;
        *reinterpret_cast<float2*>(&output[idx]) = out_vec;
    }
    else if (idx < numel) {
        // Handle edge case
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
    // Adjust block count since each thread processes 2 elements
    const int blocks = min(65535, (numel + threads * 2 - 1) / (threads * 2));

    // Ensure norm_tensor is zeroed
    cudaMemsetAsync(norm_ptr, 0, sizeof(float));

    compute_norm_kernel_optimized<<<blocks, threads>>>(input_ptr, norm_ptr, numel);
    
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrt(norm_val);

    normalize_kernel_optimized<<<blocks, threads>>>(input_ptr, output_ptr, norm_val, numel);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Frobenius norm normalization");
}