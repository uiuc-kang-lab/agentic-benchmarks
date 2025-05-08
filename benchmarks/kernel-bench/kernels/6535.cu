#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses a grid-stride loop to eliminate an early exit branch,
// which minimizes warp divergence. The inner loop is unrolled to further
// ensure uniform control flow across threads.

template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t total,
    int64_t inner_size,
    int64_t dim_size) {

    int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        // Compute outer and inner indexes from the global index
        int outer_idx = idx / inner_size;
        int inner_idx = idx % inner_size;
        int input_offset = outer_idx * dim_size * inner_size + inner_idx;

        scalar_t sum = static_cast<scalar_t>(0);
        #pragma unroll
        for (int i = 0; i < dim_size; ++i) {
            sum += input[input_offset + i * inner_size];
        }
        output[idx] = sum / static_cast<scalar_t>(dim_size);
    }
}

// Host function to launch the kernel with uniform control flow to
// minimize divergent branching among warps.

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); ++i) {
        inner_size *= sizes[i];
    }

    int64_t total = outer_size * inner_size;
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            total,
            inner_size,
            dim_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean reduction with grid-stride loop and uniform control flow (CUDA)");
}
