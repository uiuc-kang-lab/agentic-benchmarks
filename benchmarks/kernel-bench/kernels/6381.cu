#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel combining block-level and warp-level reduction

template <typename scalar_t>
__global__ void optimized_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t total_output) {

    __shared__ scalar_t shared_data[32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output) return;

    int outer_idx = idx / inner_size;
    int inner_idx = idx % inner_size;

    scalar_t sum = 0;
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    for (int i = lane; i < reduce_size; i += warpSize) {
        int64_t offset = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
        sum += input[offset];
    }

    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    if (lane == 0) {
        shared_data[warp_id] = sum;
    }

    __syncthreads();

    if (warp_id == 0) {
        sum = (threadIdx.x < blockDim.x / warpSize) ? shared_data[threadIdx.x] : 0;
        for (int offset = blockDim.x / (2 * warpSize); offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (threadIdx.x == 0) {
            output[blockIdx.x] = sum;
        }
    }
}

// Host function wrapping the kernel launch

torch::Tensor optimized_sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();
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

    sizes[dim] = 1;
    auto output = torch::empty(sizes, input.options());

    int64_t total_output = outer_size * inner_size;

    const int threads = 256;
    const int blocks = (total_output + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_sum_reduce_cuda", ([&] {
        optimized_sum_reduce_kernel<scalar_t><<<blocks, threads, threads / warpSize * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner_size,
            total_output
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_sum_reduce_cuda, "Optimized sum reduction forward (CUDA)");
}