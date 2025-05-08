#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Declare constant memory variables for read-only parameters
__constant__ int64_t g_dim_size;
__constant__ int64_t g_inner_size;

// Kernel using constant memory for frequently accessed data
template <typename scalar_t>
__global__ void max_reduce_kernel_const(
    const scalar_t* input,
    scalar_t* output,
    const int64_t outer_size
) {
    // Load constant parameters from constant memory
    int64_t dim_size = g_dim_size;
    int64_t inner_size = g_inner_size;

    int64_t total_elements = outer_size * inner_size;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int64_t outer_idx = idx / inner_size;
    int64_t inner_idx = idx % inner_size;
    int64_t start_idx = outer_idx * dim_size * inner_size + inner_idx;

    scalar_t max_val = input[start_idx];
    #pragma unroll
    for (int64_t i = 1; i < dim_size; i++) {
        scalar_t val = input[start_idx + i * inner_size];
        max_val = max(max_val, val);
    }
    output[idx] = max_val;
}

// CUDA forward function
torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    int64_t dim_size = input.size(dim);

    // Copy read-only parameters to constant memory
    cudaMemcpyToSymbol(g_dim_size, &dim_size, sizeof(int64_t));
    cudaMemcpyToSymbol(g_inner_size, &inner_size, sizeof(int64_t));

    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        max_reduce_kernel_const<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_reduce_cuda_forward, "Max reduce forward (CUDA) with constant memory");
}
