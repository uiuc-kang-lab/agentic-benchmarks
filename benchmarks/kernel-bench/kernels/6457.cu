#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

// Device function for contiguous reduction using vectorized loads when possible
// Utilizing shared memory to accumulate partial results across threads in a block
// Accelerates the reduction by performing initial reduction steps in shared memory

constexpr int WARP_SIZE = 32;

// Device function for contiguous reduction using float4 for vectorized operations
template <typename scalar_t>
__device__ inline scalar_t warp_reduce_sum(scalar_t val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Main kernel utilizing shared memory for partial reduction
// Check the memory is aligned and use float4 or double2 according to scalar type
template <typename scalar_t>
__global__ void mean_reduce_kernel(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   int64_t outer_size,
                                   int64_t dim_size,
                                   int64_t inner_size) {
    extern __shared__ scalar_t shared_data[];
    unsigned int tid = threadIdx.x;
    unsigned int block_size = blockDim.x;
    unsigned int block_id = blockIdx.x;

    if (block_id >= outer_size * inner_size) return;  // Ensure valid block processing

    int outer_idx = block_id / inner_size;
    int inner_idx = block_id % inner_size;
    int64_t offset = outer_idx * (dim_size * inner_size) + inner_idx;

    scalar_t local_sum = 0;
    if (inner_size == 1 && dim_size % 4 == 0 && (((uintptr_t)(input + offset) & 0xF) == 0)) {
        int iters = dim_size / 4;
        for (int i = tid; i < iters; i += block_size) {
            float4 v = __ldg(reinterpret_cast<const float4*>(input + offset) + i);
            local_sum += v.x + v.y + v.z + v.w;
        }
    } else {
        for (int i = tid; i < dim_size; i += block_size) {
            local_sum += __ldg(input + offset + i * inner_size);
        }
    }

    // Reduction in shared memory
    shared_data[tid] = local_sum;
    __syncthreads();

    // Reduce within block
    for (unsigned int stride = block_size / 2; stride > 32; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }

    // Final warp reduction:
    if (tid < 32) {
        shared_data[tid] += shared_data[tid + 32];
        shared_data[tid] = warp_reduce_sum(shared_data[tid]);
    }

    // Write result to output
    if (tid == 0) {
        output[block_id] = shared_data[0] / static_cast<scalar_t>(dim_size);
    }
}

/* UART buffer modifications */
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
        size_t shared_memory_size = sizeof(scalar_t) * threads;
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