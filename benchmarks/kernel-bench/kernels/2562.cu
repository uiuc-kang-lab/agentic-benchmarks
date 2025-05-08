#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// 2D-indexed CUDA kernel for ReLU activation
// This kernel maps the 1D problem domain onto a 2D grid to improve thread/block distribution
// and potentially enhance memory coalescing and occupancy when the tensor is large.

template <typename scalar_t>
__global__ void relu_2d_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int size,
    const int width) {

    // Compute 2D indices based on block and thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Convert 2D index to linear index
    int idx = row * width + col;
    
    // Ensure the linear index is within bounds
    if (idx < size) {
        scalar_t val = input[idx];
        output[idx] = val > static_cast<scalar_t>(0) ? val : static_cast<scalar_t>(0);
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();

    // Compute a 2D grid mapping: choose a width roughly equal to the square root of the total size
    const int width = static_cast<int>(ceil(sqrt(static_cast<float>(size))));
    const int height = (size + width - 1) / width;
    
    // Define block and grid dimensions for 2D indexing
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_2d_kernel", ([&] {
        relu_2d_kernel<scalar_t><<<grid, block>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            size,
            width
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with 2D indexing (CUDA)");
}
