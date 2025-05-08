#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized cumulative product kernel using grid-stride looping and loop unrolling.
// Each thread processes one cumulative product task along the specified dimension.

template <typename scalar_t>
__global__ void cumprod_kernel_optimized(
    scalar_t* output,
    const scalar_t* input,
    const int64_t total_tasks,
    const int64_t dim_size,
    const int64_t stride_val) {

    // Shared memory for caching input values
    extern __shared__ char shared_mem[];
    scalar_t* shared_data = reinterpret_cast<scalar_t*>(shared_mem);
    
    // Use grid-stride loop to cover all tasks uniformly
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_tasks; idx += gridDim.x * blockDim.x) {
        // Determine batch and column indices
        int batch = idx / stride_val;
        int col = idx % stride_val;
        // Compute the base offset for this cumulative product task
        int base = batch * (dim_size * stride_val) + col;
        
        // Load data into shared memory
        const int tile_size = min(static_cast<int64_t>(blockDim.x), dim_size);
        for (int i = threadIdx.x; i < dim_size; i += blockDim.x) {
            shared_data[i] = input[base + i * stride_val];
        }
        __syncthreads();
        
        scalar_t prod = static_cast<scalar_t>(1);
        
        // Compute cumulative product using shared memory
        #pragma unroll 4
        for (int i = 0; i < dim_size; i++) {
            prod *= shared_data[i];
            output[base + i * stride_val] = prod;
        }
        __syncthreads();
    }
}

// Host function to launch the optimized kernel

torch::Tensor cumprod_cuda_forward_optimized(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);

    // Extract tensor properties
    auto sizes = input.sizes();
    auto strides = input.strides();

    int64_t dim_size = sizes[dim];
    int64_t stride_val = strides[dim];
    int64_t numel = input.numel();

    // Each task corresponds to one cumulative product along the target dimension
    int64_t total_tasks = numel / dim_size;

    const int threads = 256;
    const int blocks = (total_tasks + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_optimized", ([&] {
        cumprod_kernel_optimized<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            total_tasks,
            dim_size,
            stride_val
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward_optimized, "Optimized cumulative product forward (CUDA)");
}
