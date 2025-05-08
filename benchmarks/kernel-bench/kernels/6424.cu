#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void uniform_warp_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size) {

    constexpr int WARP_SIZE = 32;
    const unsigned int FULL_MASK = 0xffffffff;

    // Each block handles one output element
    int out_idx = blockIdx.x;
    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;
    int64_t base = outer_idx * reduce_size * inner_size + inner_idx;

    // Warp-aligned accumulation
    scalar_t sum = 0;
    for (int i = threadIdx.x; i < reduce_size; i += WARP_SIZE) {
        sum += input[base + i * inner_size];
    }

    // Complete reduction within first warp
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    // Single warp writes final result
    if (threadIdx.x < WARP_SIZE) {
        if (threadIdx.x == 0) {
            output[out_idx] = sum;
        }
    }
}

torch::Tensor uniform_warp_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= sizes[i];
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) inner_size *= sizes[i];

    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    const int64_t num_output = outer_size * inner_size;
    const int threads = 32;  // Single warp per block
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "uniform_warp_reduce_cuda", ([&] {
        uniform_warp_reduce_kernel<scalar_t><<<num_output, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &uniform_warp_reduce_cuda, "Uniform warp reduction (CUDA)");
}