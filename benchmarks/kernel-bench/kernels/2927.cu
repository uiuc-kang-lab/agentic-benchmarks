#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel using 2D grid and block indexing for efficient thread mapping
template <typename scalar_t>
__global__ void tanh_kernel_2d(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    // Compute 2D thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate effective width based on grid's x-dimension
    int width = gridDim.x * blockDim.x;
    int index = row * width + col;

    if (index < size) {
        output[index] = tanhf(input[index]);
    }
}

// Forward function to launch the kernel
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();

    // Define block dimensions (2D): 32 x 8 = 256 threads per block
    dim3 block(32, 8);
    
    // Calculate total number of threads per block
    int threadsPerBlock = block.x * block.y;
    
    // Compute total number of blocks needed
    int totalBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    
    // Organize blocks in a 2D grid for better occupancy
    int grid_x = static_cast<int>(ceil(sqrt(static_cast<float>(totalBlocks))));
    int grid_y = (totalBlocks + grid_x - 1) / grid_x;
    dim3 grid(grid_x, grid_y);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_2d", ([&] {
        tanh_kernel_2d<scalar_t><<<grid, block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with 2D indexing (CUDA)");
}
