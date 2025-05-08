#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define constant memory for frequently accessed data
#include <type_traits>
#include <cuda_fp16.h>

__constant__ float constant_input_float[1024];
__constant__ __half constant_input_half[1024];

// Cumulative product kernel using constant memory for input storage
// Note: Ensuring data fits into constant memory limits is crucial

template <typename scalar_t>
__global__ void cumprod_kernel_constant(
    scalar_t* output,
    const scalar_t* input,
    const int64_t numel,
    const int64_t dim_size,
    const int64_t stride) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / stride;
    const int in_idx = idx % stride;

    if (idx < numel / dim_size) {
        scalar_t product = 1;
        for (int i = 0; i < dim_size; i++) {
            const int64_t curr_idx = batch_idx * (stride * dim_size) + i * stride + in_idx;
            // Access input from constant memory
            product *= constant_input[curr_idx];
            output[curr_idx] = product;
        }
    }
}

torch::Tensor cumprod_cuda_forward_constant(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    // Get tensor properties
    auto sizes = input.sizes();
    auto strides = input.strides();
    
    // Calculate dimension properties
    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();
    
    // Calculate total number of elements to process
    int64_t total_threads = numel / dim_size;
    
    // Transfer input to constant memory
    cudaMemcpyToSymbol(constant_input, input.data_ptr<scalar_t>(), numel * sizeof(scalar_t));

    // CUDA kernel launch parameters
    const int threads = 256;
    const int blocks = (total_threads + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_constant", ([&] {
        cumprod_kernel_constant<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            numel,
            dim_size,
            stride
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward_constant, "Cumulative product forward using constant memory (CUDA)");
}
