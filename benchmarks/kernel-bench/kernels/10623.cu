#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// This kernel uses a grid-stride loop to ensure that all threads follow the same loop structure,
// minimizing warp divergence by removing conditional branches inside the loop.
// Each thread processes a cumulative product task along the specified dimension uniformly.

template <typename scalar_t>
__global__ void cumprod_kernel_uniform(
    scalar_t* output,
    const scalar_t* input,
    const int64_t total_tasks,
    const int64_t dim_size,
    const int64_t stride) {

    // Use grid-stride loop to process every valid task without extra divergent conditionals
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_tasks; idx += blockDim.x * gridDim.x) {
        int batch_idx = idx / stride;
        int in_idx = idx % stride;
        int start_idx = batch_idx * (dim_size * stride) + in_idx;
        
        scalar_t product = 1;
        #pragma unroll
        for (int i = 0; i < dim_size; i++) {
            int curr_idx = start_idx + i * stride;
            product *= input[curr_idx];
            output[curr_idx] = product;
        }
    }
}


// Host function to launch the kernel using uniform control flow

torch::Tensor cumprod_cuda_forward_uniform(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);

    auto sizes = input.sizes();
    auto strides = input.strides();

    int64_t dim_size = sizes[dim];
    int64_t stride = strides[dim];
    int64_t numel = input.numel();

    // Each task corresponds to computing the cumulative product along one segment
    int64_t total_tasks = numel / dim_size;

    const int threads = 256;
    const int blocks = (total_tasks + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "cumprod_cuda_uniform", ([&] {
        cumprod_kernel_uniform<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            total_tasks,
            dim_size,
            stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &cumprod_cuda_forward_uniform, "Cumulative product forward with uniform control flow (CUDA)");
}
