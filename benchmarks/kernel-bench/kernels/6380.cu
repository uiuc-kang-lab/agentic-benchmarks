#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel combining direct reduction and atomic operations for large reduction sizes
// Uses shared memory and minimizes atomic operations

template <typename scalar_t>
__global__ void optimized_sum_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t reduce_size,
    int64_t inner_size,
    int64_t n_out) {

    int out_idx = blockIdx.x; 
    if (out_idx >= n_out) return;

    int outer_idx = out_idx / inner_size;
    int inner_idx = out_idx % inner_size;

    int threadsInBlock = blockDim.x;
    int seg = blockIdx.y; 
    int chunks = gridDim.y;
    int chunk_size = (reduce_size + chunks - 1) / chunks;
    int start = seg * chunk_size;
    int end = start + chunk_size;
    if (end > reduce_size) end = reduce_size;

    scalar_t sum = 0;
    int64_t base = outer_idx * reduce_size * inner_size + inner_idx;

    for (int i = start + threadIdx.x; i < end; i += threadsInBlock) {
        int64_t offset = base + i * inner_size;
        sum += input[offset];
    }

    extern __shared__ char smem[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(smem);
    shared[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int stride = threadsInBlock / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(&output[out_idx], shared[0]);
    }
}

// Host function to launch the optimized kernel

torch::Tensor optimized_sum_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t reduce_size = sizes[dim];

    int64_t outer = 1;
    for (int i = 0; i < dim; i++) {
        outer *= sizes[i];
    }
    int64_t inner = 1;
    for (size_t i = dim + 1; i < sizes.size(); i++) {
        inner *= sizes[i];
    }
    sizes[dim] = 1;

    auto output = torch::zeros(sizes, input.options());
    int64_t n_out = outer * inner;

    const int threads = 256;
    int blocks_y = (reduce_size + threads - 1) / threads;
    dim3 grid(n_out, blocks_y);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_sum_reduce_cuda", ([&] {
        optimized_sum_reduce_kernel<scalar_t><<<grid, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            reduce_size,
            inner,
            n_out
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_sum_reduce_cuda, "Optimized sum reduction forward (CUDA)");
}
