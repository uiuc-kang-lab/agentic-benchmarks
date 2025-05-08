#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Inline device function for GELU activation
__device__ __forceinline__ float gelu_act(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    // Use tanh intrinsic for better performance
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Optimized kernel using shared memory tiling and loop unrolling
__global__ void gelu_kernel_tile_inline(const float* __restrict__ x,
                                          float* __restrict__ y, int n) {
    const int unroll = 4;
    int tileSize = blockDim.x * unroll;
    int base = blockIdx.x * tileSize;

    extern __shared__ float tile[];
    int tid = threadIdx.x;

    // Cooperative loading of a tile from global memory to shared memory
    #pragma unroll
    for (int i = 0; i < unroll; i++) {
        int idx = base + tid + i * blockDim.x;
        if (idx < n) {
            tile[tid + i * blockDim.x] = x[idx];
        }
    }
    __syncthreads();

    // Compute GELU using the data loaded into shared memory
    #pragma unroll
    for (int i = 0; i < unroll; i++) {
        int idx = base + tid + i * blockDim.x;
        if (idx < n) {
            float val = tile[tid + i * blockDim.x];
            y[idx] = gelu_act(val);
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
    const int unroll = 4;
    int tileSize = threads * unroll;
    int blocks = (n + tileSize - 1) / tileSize;

    // Allocate shared memory: one tile per block
    size_t sharedMemSize = tileSize * sizeof(float);

    gelu_kernel_tile_inline<<<blocks, threads, sharedMemSize>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Optimized GELU forward CUDA implementation using shared memory tiling and inline device function");
}
