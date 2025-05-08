#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function for GELU activation
__device__ __forceinline__ float gelu_activation(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Kernel that uses warp-level primitives for efficient computation
__global__ void gelu_kernel_warp_optimized(const float* __restrict__ x, float* __restrict__ y, int n) {
    extern __shared__ float shared_x[];
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int gid = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (gid < n) {
        shared_x[tid] = x[gid];
    }
    __syncthreads();

    // Perform GELU activation using warp-level primitives
    if (gid < n) {
        float val = shared_x[tid];
        float result = gelu_activation(val);

        // Use warp shuffle to propagate results within the warp
        for (int offset = 16; offset > 0; offset /= 2) {
            result += __shfl_down_sync(0xFFFFFFFF, result, offset);
        }

        // Write the result back to global memory
        if (lane_id == 0) {
            y[blockIdx.x * 32 + warp_id] = result;
        }
    }
}

// Host function to launch the kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    size_t shared_mem_size = threads * sizeof(float);
    
    gelu_kernel_warp_optimized<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Warp-optimized GELU forward CUDA implementation");
}