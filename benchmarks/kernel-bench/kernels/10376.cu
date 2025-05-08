#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Optimized vectorized kernel using stride loop and shared memory
__global__ void gelu_kernel_optimized(const float4* __restrict__ x, float4* __restrict__ y, int vec_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Constants stored in shared memory for faster access
    __shared__ float constants[2];
    if (threadIdx.x == 0) {
        constants[0] = 0.7978845608f;  // sqrt_2_over_pi
        constants[1] = 0.044715f;      // coeff
    }
    __syncthreads();
    
    const float sqrt_2_over_pi = constants[0];
    const float coeff = constants[1];
    
    #pragma unroll
    for (int i = idx; i < vec_size; i += stride) {
        float4 v = __ldg(&x[i]);
        float4 out;
        
        // Process all components in parallel using compiler optimizations
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float xj = (&v.x)[j];
            float xj_cubed = xj * xj * xj;
            float inner = (xj + coeff * xj_cubed) * sqrt_2_over_pi;
            (&out.x)[j] = 0.5f * xj * (1.0f + tanhf(inner));
        }
        
        y[i] = out;
    }
}

// Optimized scalar kernel for remainder using shared memory
__global__ void gelu_kernel_scalar_optimized(const float* __restrict__ x, float* __restrict__ y, 
                                           int n, int vec_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Constants in shared memory
    __shared__ float constants[2];
    if (threadIdx.x == 0) {
        constants[0] = 0.7978845608f;
        constants[1] = 0.044715f;
    }
    __syncthreads();
    
    const float sqrt_2_over_pi = constants[0];
    const float coeff = constants[1];

    #pragma unroll
    for (int i = idx + vec_offset; i < n; i += stride) {
        float xi = __ldg(&x[i]);
        float xi_cubed = xi * xi * xi;
        float inner = (xi + coeff * xi_cubed) * sqrt_2_over_pi;
        y[i] = 0.5f * xi * (1.0f + tanhf(inner));
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int size = x.numel();
    
    int vec_size = size / 4;
    int remainder = size % 4;
    
    // Optimize grid size based on SM count
    int device_id;
    cudaGetDevice(&device_id);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    
    const int threads = 256;
    // Calculate optimal number of blocks based on occupancy and data size
    int max_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        gelu_kernel_optimized,
        threads,
        0  // Dynamic shared memory size
    );
    
    // Calculate grid size to cover data with optimal occupancy
    int blocks = min(
        sm_count * max_blocks_per_sm,
        (size + threads - 1) / threads
    );

    if (vec_size > 0) {
        const float4* x_vec = reinterpret_cast<const float4*>(x.data_ptr<float>());
        float4* y_vec = reinterpret_cast<float4*>(y.data_ptr<float>());
        gelu_kernel_optimized<<<blocks, threads>>>(x_vec, y_vec, vec_size);
    }

    if (remainder > 0) {
        int offset = vec_size * 4;
        gelu_kernel_scalar_optimized<<<blocks, threads>>>(
            x.data_ptr<float>(), y.data_ptr<float>(), size, offset);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Optimized GELU CUDA implementation");
}