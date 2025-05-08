#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute the sum over the reduction dimension
template <typename scalar_t>
__device__ inline scalar_t compute_sum(const scalar_t* __restrict__ input,
                                         int outer_idx,
                                         int inner_idx,
                                         int dim_size,
                                         int inner_size) {
    int offset = outer_idx * dim_size * inner_size + inner_idx;
    scalar_t sum = static_cast<scalar_t>(0);
    #pragma unroll
    for (int i = 0; i < dim_size; ++i) {
        sum += input[offset + i * inner_size];
    }
    return sum;
}

// Device function to compute the mean using the computed sum
template <typename scalar_t>
__device__ inline scalar_t compute_mean(const scalar_t* __restrict__ input,
                                          int outer_idx,
                                          int inner_idx,
                                          int dim_size,
                                          int inner_size) {
    scalar_t sum = compute_sum<scalar_t>(input, outer_idx, inner_idx, dim_size, inner_size);
    return sum / static_cast<scalar_t>(dim_size);
}

// Kernel using a grid-stride loop that calls modular device functions
template <typename scalar_t>
__global__ void mean_reduce_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      int outer_size,
                                      int dim_size,
                                      int inner_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total = outer_size * inner_size;
    
    for (int idx = tid; idx < total; idx += stride) {
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;
        output[idx] = compute_mean<scalar_t>(input, outer_idx, inner_idx, dim_size, inner_size);
    }
}

// Host function to prepare data and launch the kernel
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    // Obtain sizes and compute dimensions for reduction
    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Create output tensor with reduced dimension removed
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    // Configure CUDA launch parameters
    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            outer_size,
            dim_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Modular Mean Reduction (CUDA)");
}
