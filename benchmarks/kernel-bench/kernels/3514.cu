#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device helper: define an inline exponential function for float and double
template <typename scalar_t>
__device__ inline scalar_t my_exp(scalar_t x);

template <>
__device__ inline float my_exp<float>(float x) {
    return expf(x);
}

template <>
__device__ inline double my_exp<double>(double x) {
    return exp(x);
}

// CUDA kernel with 2D thread and block indexing for efficient mapping
// This kernel maps the flat 1D input array to a 2D grid of threads to enhance occupancy and performance

template <typename scalar_t>
__global__ void selu_kernel_2d_indexed(const scalar_t* __restrict__ input,
                                       scalar_t* __restrict__ output,
                                       size_t numel) {
    // Compute global thread index using 2D block and grid indexing
    int tIdx = threadIdx.x + threadIdx.y * blockDim.x;
    int bIdx = blockIdx.x + blockIdx.y * gridDim.x;
    size_t global_idx = bIdx * (blockDim.x * blockDim.y) + tIdx;
    
    // Total number of threads in the grid
    size_t total_threads = gridDim.x * gridDim.y * blockDim.x * blockDim.y;

    for (size_t i = global_idx; i < numel; i += total_threads) {
        scalar_t x = __ldg(&input[i]);
        scalar_t res = (x > static_cast<scalar_t>(0))
                           ? x
                           : static_cast<scalar_t>(1.67326324235437728481) * (my_exp(x) - static_cast<scalar_t>(1));
        output[i] = static_cast<scalar_t>(1.05070098735548049342) * res;
    }
}

// Host function that prepares the 2D grid and block dimensions and launches the kernel

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");

    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    // Define 2D block dimensions
    const int block_dim_x = 32, block_dim_y = 8; // 256 threads per block
    dim3 threads(block_dim_x, block_dim_y);

    // Compute the number of blocks needed given the total number of elements
    int block_size = block_dim_x * block_dim_y;
    int blocks_needed = (numel + block_size - 1) / block_size;

    // Arrange blocks in a 2D grid: use a near-square grid layout
    int grid_dim_x = static_cast<int>(ceil(sqrt((double)blocks_needed)));
    int grid_dim_y = (blocks_needed + grid_dim_x - 1) / grid_dim_x;
    dim3 blocks(grid_dim_x, grid_dim_y);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "selu_forward_cuda_2d", ([&] {
        const scalar_t* input_ptr = input.data_ptr<scalar_t>();
        scalar_t* output_ptr = output.data_ptr<scalar_t>();
        selu_kernel_2d_indexed<scalar_t><<<blocks, threads>>>(input_ptr, output_ptr, numel);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward with 2D Thread Indexing (CUDA)");
}
