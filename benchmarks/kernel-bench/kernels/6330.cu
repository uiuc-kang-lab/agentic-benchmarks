#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel: Each block handles one outer index for coalesced access in the inner dimension.
template <typename scalar_t>
__global__ void sum_reduce_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t reduce_size,
    int64_t inner_size) {

    // Each block corresponds to one outer index
    int outer_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Compute base offset for this outer index
    int64_t base_offset = outer_idx * reduce_size * inner_size;
    int64_t out_offset = outer_idx * inner_size;

    // Each thread processes multiple inner indices, striding by blockDim.x
    for (int inner_idx = tid; inner_idx < inner_size; inner_idx += blockDim.x) {
        scalar_t sum = 0;
        // Reduction along the reduce dimension
        for (int i = 0; i < reduce_size; i++) {
            sum += input[base_offset + i * inner_size + inner_idx];
        }
        output[out_offset + inner_idx] = sum;
    }
}

// CUDA wrapper function
torch::Tensor sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) {
        dim += input.dim();
    }

    // Get sizes and dimensions
    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // The output tensor has the reduction dimension set to 1
    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    // Experiment with block sizes based on the inner_size:
    // Choose from candidate block sizes: 32, 64, 128, 256, 512
    int block_size;
    if (inner_size >= 512) {
        block_size = 512;
    } else if (inner_size >= 256) {
        block_size = 256;
    } else if (inner_size >= 128) {
        block_size = 128;
    } else if (inner_size >= 64) {
        block_size = 64;
    } else if (inner_size >= 32) {
        block_size = 32;
    } else {
        block_size = inner_size; // For very small inner_size
    }

    // Each block handles one outer slice
    int blocks = outer_size;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda", ([&] {
        sum_reduce_kernel<scalar_t><<<blocks, block_size>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda, "Sum reduction forward (CUDA)");
}
