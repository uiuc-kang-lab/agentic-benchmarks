#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Warp-level reduction using only shuffle operations
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduction using cascading warp reductions without shared memory
__device__ __forceinline__ float blockReduceSum(float val) {
    const int lid = threadIdx.x % WARP_SIZE;
    const int wid = threadIdx.x / WARP_SIZE;
    const int nWarps = BLOCK_SIZE / WARP_SIZE;
    
    // First warp reduction
    val = warpReduceSum(val);
    
    // Get the partial sum from each warp
    float warp_sum = __shfl_sync(0xffffffff, val, 0);
    
    // Final reduction in the first warp
    if (wid == 0) {
        if (lid >= nWarps) warp_sum = 0.0f;
        warp_sum = warpReduceSum(warp_sum);
    }
    
    return __shfl_sync(0xffffffff, warp_sum, 0);
}

// Compute kernel using vectorized loads and warp reductions
__global__ __launch_bounds__(BLOCK_SIZE) void compute_norm_kernel(const float4* __restrict__ input4, float* norm_out, int numel4) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time
    for (int i = idx; i < numel4; i += stride) {
        float4 v = input4[i];
        sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    
    // Block-wide reduction using warp primitives
    sum = blockReduceSum(sum);
    
    if (threadIdx.x == 0) {
        atomicAdd(norm_out, sum);
    }
}

// Normalize kernel using vectorized operations
__global__ void normalize_kernel(const float4* __restrict__ input4, 
                               float4* __restrict__ output4,
                               float norm,
                               int numel4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float inv_norm = 1.0f / norm;
    
    if (idx < numel4) {
        float4 v = input4[idx];
        v.x *= inv_norm;
        v.y *= inv_norm;
        v.z *= inv_norm;
        v.w *= inv_norm;
        output4[idx] = v;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on GPU");
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(input);
    auto norm_tensor = torch::zeros({1}, input.options());

    const int numel = input.numel();
    const int numel4 = numel / 4;
    const float4* input4_ptr = reinterpret_cast<const float4*>(input.data_ptr<float>());
    float4* output4_ptr = reinterpret_cast<float4*>(output.data_ptr<float>());
    float* norm_ptr = norm_tensor.data_ptr<float>();

    const int threads = BLOCK_SIZE;
    const int blocks = min(65535, (numel4 + threads - 1) / threads);

    // Compute norm
    compute_norm_kernel<<<blocks, threads>>>(input4_ptr, norm_ptr, numel4);
    
    // Handle remaining elements if any
    if (numel % 4 != 0) {
        const float* input_ptr = input.data_ptr<float>();
        float host_sum = 0.0f;
        for (int i = numel4 * 4; i < numel; i++) {
            float val = input_ptr[i];
            host_sum += val * val;
        }
        float device_sum = 0.0f;
        cudaMemcpy(&device_sum, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
        device_sum += host_sum;
        cudaMemcpy(norm_ptr, &device_sum, sizeof(float), cudaMemcpyHostToDevice);
    }

    // Get final norm
    float norm_val;
    cudaMemcpy(&norm_val, norm_ptr, sizeof(float), cudaMemcpyDeviceToHost);
    norm_val = sqrtf(norm_val);

    // Normalize
    normalize_kernel<<<blocks, threads>>>(input4_ptr, output4_ptr, norm_val, numel4);

    // Handle remaining elements
    if (numel % 4 != 0) {
        const float* input_ptr = input.data_ptr<float>();
        float* output_ptr = output.data_ptr<float>();
        for (int i = numel4 * 4; i < numel; i++) {
            output_ptr[i] = input_ptr[i] / norm_val;
        }
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized Frobenius norm normalization");
}