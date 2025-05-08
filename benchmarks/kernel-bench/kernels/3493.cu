#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t gelu_function(scalar_t x) {
    constexpr float sqrt_2_inv = 0.7071067811865475f;
    return x * scalar_t(0.5) * (scalar_t(1) + erf(x * scalar_t(sqrt_2_inv)));
}

template <typename scalar_t, int VEC_SIZE>
__global__ void gelu_kernel_2d_optimized(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    size_t total_elements
) {
    const int block_stride = gridDim.x * blockDim.x * VEC_SIZE;
    const int thread_offset = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * VEC_SIZE + threadIdx.x * VEC_SIZE;

    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        int elem_idx = thread_offset + i;
        if (elem_idx < total_elements) {
            output[elem_idx] = gelu_function(input[elem_idx]);
        }
    }
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA tensor");
    auto y = torch::empty_like(x);
    const size_t numel = x.numel();

    constexpr int VEC_SIZE = sizeof(float) == sizeof(scalar_t) ? 4 : 2;
    const int threads = 128;
    const int blocks_x = 132 * 4;  // Match H100's 132 SMs with 4 blocks per SM
    const int blocks_y = (numel + (blocks_x * threads * VEC_SIZE) - 1) / (blocks_x * threads * VEC_SIZE);
    dim3 grid_dim(blocks_x, std::min(blocks_y, 65535));

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "gelu_2d_kernel", [&] {
        gelu_kernel_2d_optimized<scalar_t, VEC_SIZE><<<grid_dim, threads>>>(
            x.data_ptr<scalar_t>(),
            y.data_ptr<scalar_t>(),
            numel
        );
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU forward with 2D grid optimization (CUDA)");
}