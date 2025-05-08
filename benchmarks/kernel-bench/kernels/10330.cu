#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function for GELU activation computation
__device__ __forceinline__ float gelu_compute(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Main kernel processing the bulk of the data
// This kernel assumes that the number of elements (main_n) is a multiple of (blockDim.x * unroll)
__global__ void gelu_kernel_main(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int main_n) {
    const int unroll = 4;
    int base = blockIdx.x * blockDim.x * unroll;
    int tid = threadIdx.x;

    extern __shared__ float tile[]; // shared memory for blockDim.x * unroll floats

    // Cooperative loading without boundary checks since all indices are valid
    #pragma unroll
    for (int i = 0; i < unroll; i++) {
        int index = base + tid + i * blockDim.x;
        tile[tid + i * blockDim.x] = x[index];
    }
    __syncthreads();

    // Compute GELU activation from shared memory; no divergent conditionals here
    #pragma unroll
    for (int i = 0; i < unroll; i++) {
        int index = base + tid + i * blockDim.x;
        float val = tile[tid + i * blockDim.x];
        y[index] = gelu_compute(val);
    }
}

// Tail kernel to process any remaining elements that don't make up a full block
__global__ void gelu_kernel_tail(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int n, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_idx = offset + idx;
    if (global_idx < n) {
        float val = x[global_idx];
        y[global_idx] = gelu_compute(val);
    }
}

// Host function: splits the workload to minimize warp divergence
// The main kernel is launched over a contiguous segment where boundary checks are not required,
// thus ensuring uniform control flow in warps. The tail kernel handles the remainder with minimal divergence.

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    const int unroll = 4;
    const int work_per_block = threads * unroll;

    // Compute main_n as the largest multiple of work_per_block less than or equal to n
    int main_n = (n / work_per_block) * work_per_block;
    int tail = n - main_n;

    // Launch main kernel for the majority of data with no divergent branches
    if (main_n > 0) {
        int main_blocks = main_n / work_per_block;
        size_t shared_mem_size = threads * unroll * sizeof(float);
        gelu_kernel_main<<<main_blocks, threads, shared_mem_size>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            main_n
        );
    }

    // Launch tail kernel for any remaining elements
    if (tail > 0) {
        int tail_blocks = (tail + threads - 1) / threads;
        gelu_kernel_tail<<<tail_blocks, threads>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            n,
            main_n
        );
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation minimizing warp divergence");
}
