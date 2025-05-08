#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Unified CUDA kernel that combines efficient warp-level and block-level reduction
template <typename scalar_t>
__global__ void efficient_mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    // Allocate shared memory for storing per-block sums
    __shared__ scalar_t shared_data[256];  // Fixed size shared memory allocation

    // Identify thread and block indices
    const int tid = threadIdx.x;
    const int output_idx = blockIdx.x;
    if (output_idx >= outer_size * inner_size) return;

    const int outer_idx = output_idx / inner_size;
    const int inner_idx = output_idx % inner_size;
    const int base_idx = outer_idx * dim_size * inner_size + inner_idx;

    // Load and sum elements directly into registers with loop unrolling
    scalar_t sum = 0;
    #pragma unroll 4
    for (int i = tid; i < dim_size; i += blockDim.x) {
        sum += input[base_idx + i * inner_size];
    }

    // Store the sum in shared memory
    shared_data[tid] = sum;
    __syncthreads();

    // Perform block-level reduction using shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shared_data[tid] += shared_data[tid + offset];
        }
        __syncthreads();
    }

    // Write the final mean value
    if (tid == 0) {
        output[output_idx] = shared_data[0] / static_cast<scalar_t>(dim_size);
    }
}

// Host function to prepare and launch the CUDA kernel
torch::Tensor efficient_mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    const int threads = 256;
    const int blocks = outer_size * inner_size;
    const int shared_mem_size = threads * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "efficient_mean_reduce_cuda", ([&] {
        efficient_mean_reduce_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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
    m.def("forward", &efficient_mean_reduce_cuda, "Efficient Mean reduction (CUDA)");
}