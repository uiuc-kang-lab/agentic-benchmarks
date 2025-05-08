#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Device function to compute softplus in a numerically stable way
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_softplus(scalar_t x) {
    // Use faster intrinsic functions for exponential and logarithm
    const scalar_t THRESHOLD = 20.0;
    if (x > THRESHOLD) {
        return x;
    } else if (x < -THRESHOLD) {
        return __expf(x);  // Fast intrinsic exp for float
    } else {
        return log1pf(__expf(x));  // Fast intrinsic functions
    }
}

// Kernel using 2D block and grid indexing
template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    // Compute a unique thread index using 2D block and grid dimensions
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int block_id  = blockIdx.y * gridDim.x + blockIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    int idx = block_id * threads_per_block + thread_id;
    
    // Compute total number of threads across the grid
    int total_threads = gridDim.x * gridDim.y * threads_per_block;
    
    // Process elements in a grid-stride loop
    while (idx < size) {
        const scalar_t x = input[idx];
        output[idx] = compute_softplus(x);
        idx += total_threads;
    }
}

// CUDA forward function
torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Define a 2D block configuration: 16x16 = 256 threads per block
    dim3 block(16, 16);
    
    // Calculate total blocks needed and organize them into a 2D grid
    int blocks_needed = (size + (block.x * block.y) - 1) / (block.x * block.y);
    int grid_dim = static_cast<int>(ceil(sqrt(static_cast<double>(blocks_needed))));
    dim3 grid(grid_dim, grid_dim);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
