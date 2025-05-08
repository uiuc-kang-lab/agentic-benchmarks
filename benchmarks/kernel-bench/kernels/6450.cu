#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

// Reduce sum with shared memory and parallel reduction in block
__inline__ __device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel for reducing using shared memory and warp shuffle
template <typename scalar_t>
__global__ void mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size) {

    extern __shared__ float shared_data[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    if (tid >= outer_size * inner_size) return;

    int outer_idx = tid / inner_size;
    int inner_idx = tid % inner_size;
    int64_t offset = outer_idx * dim_size * inner_size + inner_idx;

    float local_sum = 0.0f;
    for (int i = 0; i < dim_size; i++) {
        local_sum += input[offset + i * inner_size];
    }

    // Perform warp reduction
    local_sum = warp_reduce_sum(local_sum);

    // Only the first thread of each warp writes to shared memory
    if (lane == 0) shared_data[warp_id] = local_sum;

    __syncthreads();

    // Final reduce within the block
    float block_sum = (threadIdx.x < blockDim.x / warpSize) ? shared_data[lane] : 0;
    if (warp_id == 0) {
        block_sum = warp_reduce_sum(block_sum);
        if (threadIdx.x == 0) output[tid] = block_sum / dim_size;
    }
}

// Host function launching the kernel
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    auto sizes = input.sizes().vec();
    int64_t dim_size = sizes[dim];

    // Calculate outer and inner sizes
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (int i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Prepare output tensor by removing the reduction dimension
    sizes.erase(sizes.begin() + dim);
    auto output = torch::empty(sizes, input.options());

    const int threads = 256;
    const int blocks = (outer_size * inner_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
        mean_reduce_kernel<scalar_t><<<blocks, threads, threads / warpSize * sizeof(float)>>>(
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
    m.def("forward", &mean_reduce_cuda, "Mean reduction with shared memory and warp shuffle (CUDA)");
}