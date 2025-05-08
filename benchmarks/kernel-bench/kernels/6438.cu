#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

// Kernel function for mean reduction using shared memory and warp-level primitives
template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    extern __shared__ scalar_t shared_data[];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    if (global_tid >= outer_size * inner_size) return;

    int outer_idx = global_tid / inner_size;
    int inner_idx = global_tid % inner_size;
    int64_t offset = outer_idx * dim_size * inner_size + inner_idx;

    scalar_t sum = 0;
    for (int i = tid; i < dim_size; i += blockDim.x) {
        sum += __ldg(input + offset + i * inner_size);
    }
    shared_data[tid] = sum;
    __syncthreads();

    // Reduce within block using shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Use warp-level primitives to finalize reduction
    if (tid < 32) {
        scalar_t warp_sum = shared_data[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, offset);
        }
        if (tid == 0) {
            output[blockIdx.x] = warp_sum / static_cast<scalar_t>(dim_size);
        }
    }
}

// Host function to launch the kernel
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

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

    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;
    int shared_memory_size = threads * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads, shared_memory_size>>>(
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
    m.def("forward", &mean_reduce_cuda, "Mean reduction (CUDA)");
}
