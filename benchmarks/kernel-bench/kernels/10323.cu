#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Manually unrolled GELU CUDA kernel using shared memory
__global__ void gelu_kernel_manual_unroll(const float* __restrict__ x, float* __restrict__ y, int n) {
    extern __shared__ float shared_x[];
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int base = blockIdx.x * blockSize * 4; // unroll factor of 4

    // Compute global indices for each unrolled iteration
    int idx0 = base + tid;
    int idx1 = base + tid + blockSize;
    int idx2 = base + tid + 2 * blockSize;
    int idx3 = base + tid + 3 * blockSize;

    // Manually unroll loading data into shared memory
    if (idx0 < n) {
        shared_x[tid] = x[idx0];
    }
    if (idx1 < n) {
        shared_x[tid + blockSize] = x[idx1];
    }
    if (idx2 < n) {
        shared_x[tid + 2 * blockSize] = x[idx2];
    }
    if (idx3 < n) {
        shared_x[tid + 3 * blockSize] = x[idx3];
    }

    __syncthreads();

    // Constants for GELU computation
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    // Manually unroll computation for each element
    if (idx0 < n) {
        float xi = shared_x[tid];
        float x_cubed = xi * xi * xi;
        float inner = (xi + coeff * x_cubed) * sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[idx0] = 0.5f * xi * (1.0f + tanh_val);
    }
    if (idx1 < n) {
        float xi = shared_x[tid + blockSize];
        float x_cubed = xi * xi * xi;
        float inner = (xi + coeff * x_cubed) * sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[idx1] = 0.5f * xi * (1.0f + tanh_val);
    }
    if (idx2 < n) {
        float xi = shared_x[tid + 2 * blockSize];
        float x_cubed = xi * xi * xi;
        float inner = (xi + coeff * x_cubed) * sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[idx2] = 0.5f * xi * (1.0f + tanh_val);
    }
    if (idx3 < n) {
        float xi = shared_x[tid + 3 * blockSize];
        float x_cubed = xi * xi * xi;
        float inner = (xi + coeff * x_cubed) * sqrt_2_over_pi;
        float tanh_val = tanhf(inner);
        y[idx3] = 0.5f * xi * (1.0f + tanh_val);
    }
}

// Host function to launch the manually unrolled GELU kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    const int unroll_factor = 4;
    int blocks = (n + threads * unroll_factor - 1) / (threads * unroll_factor);
    size_t shared_mem_bytes = threads * unroll_factor * sizeof(float);

    gelu_kernel_manual_unroll<<<blocks, threads, shared_mem_bytes>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation with manual loop unrolling");
}
