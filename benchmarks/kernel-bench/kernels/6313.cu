#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sum_reduce_shared_memory_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t numel,
    int64_t reduce_size,
    int64_t outer_size,
    int64_t inner_size) {
    extern __shared__ scalar_t shared_data[];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    if (tid >= outer_size * inner_size) return;

    const int outer_idx = tid / inner_size;
    const int inner_idx = tid % inner_size;

    scalar_t sum = 0;
    const int64_t base_idx = outer_idx * reduce_size * inner_size + inner_idx;

    // Perform reduction along the specified dimension
    for (int i = threadIdx.x; i < reduce_size; i += blockDim.x) {
        sum += input[base_idx + i * inner_size];
    }

    // Store partial results in shared memory
    shared_data[threadIdx.x] = sum;
    __syncthreads();

    // Reduce within block using shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // Use warp-level primitives for the final reduction
    if (warp_id == 0) {
        sum = (lane < blockDim.x / 32) ? shared_data[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        output[outer_idx * inner_size + inner_idx] = sum;
    }
}

torch::Tensor sum_reduce_cuda_optimized(torch::Tensor input, int64_t dim) {
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

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sum_reduce_cuda_optimized", ([&] {
        sum_reduce_shared_memory_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel(),
            reduce_size,
            outer_size,
            inner_size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sum_reduce_cuda_optimized, "Sum reduction optimized forward (CUDA)");
}
