#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel optimized with shared memory and necessary synchronization
// Utilize shared memory if the amount of data to be processed per block
// justifies its use, which is often effective in reducing global memory 
// accesses and speeding up memory-bound operations.

// Adjust this according to the expected input sizes and their batch
// to calibrate shared memory usage effectively.
#define TILE_SIZE 1024

// CUDA kernel with shared memory for efficient memory access
template <typename scalar_t>
__global__ void optimized_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                             scalar_t* __restrict__ output,
                                             size_t numel) {
    __shared__ scalar_t tile[TILE_SIZE];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int localIdx = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Load input elements into shared memory
    if (idx < numel) {
        tile[localIdx] = input[idx];
    }

    // Wait for all threads to complete data loading
    __syncthreads();

    // Compute on elements in shared memory
    if (idx < numel) {
        const scalar_t add_const = static_cast<scalar_t>(3);
        const scalar_t div_const = static_cast<scalar_t>(1)/static_cast<scalar_t>(6);

        scalar_t x = tile[localIdx];
        scalar_t y = (x + add_const) * div_const;
        y = fminf(fmaxf(y, 0.0f), 1.0f);
        output[idx] = y;
    }
    // Ensure all threads in the block have finished execution before
    // moving to the next section of the grid
    __syncthreads();
}

// Host function to call the optimized kernel
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    const int threads = TILE_SIZE;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_hardsigmoid_cuda", ([&] {
        optimized_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized HardSigmoid activation forward (CUDA) with shared memory");
}