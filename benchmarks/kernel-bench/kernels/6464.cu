/*
Combined Mean Reduction CUDA Kernel
This file implements a unified CUDA kernel for computing the mean along a given dimension. Depending on the reduction size and memory layout, it selects between a no-atomic (single block per output) version and a multi-block atomic version. Both versions exploit vectorized loads when the reduction dimension is contiguous (i.e., inner_size==1) and alignment conditions are met for float (using float4) or double (using double2).
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

// Define block configuration
#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4

// Combined Mean Reduction Kernel - No Atomic Version
// Each block computes one output element using a tiled reduction and performs the mean division.

template <typename scalar_t>
__global__ void combined_mean_reduce_noatomic(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int L,       // length of reduction dimension
    int stride,  // stride between reduction elements (inner_size)
    int N        // total number of output elements
) {
    int out_idx = blockIdx.x;  // each block handles one output element (flattened)
    if (out_idx >= N) return;

    // Decode the flat output index into outer and inner indices
    int outer_idx = out_idx / stride;
    int inner_idx = out_idx % stride;
    // Base offset for the current output element's reduction slice
    int base_offset = outer_idx * (L * stride) + inner_idx;

    scalar_t local_sum = static_cast<scalar_t>(0);

    // Tiled loop: divide the L elements among threads in the block
    int tile_size = (L + blockDim.x - 1) / blockDim.x;
    int start_idx = threadIdx.x * tile_size;
    int end_idx = (start_idx + tile_size < L) ? (start_idx + tile_size) : L;

    if (stride == 1) { // contiguous reduction
        if constexpr (std::is_same<scalar_t, float>::value) {
            // Use vectorized loads if the tile length is a multiple of 4 and aligned
            if ((end_idx - start_idx) > 0 && ((end_idx - start_idx) % 4 == 0) &&
                (((uintptr_t)(input + base_offset + start_idx) & 0xF) == 0)) {
                int num_vec = (end_idx - start_idx) / 4;
                for (int i = 0; i < num_vec; i++) {
                    float4 val = __ldg(reinterpret_cast<const float4*>(input + base_offset + start_idx) + i);
                    local_sum += val.x + val.y + val.z + val.w;
                }
            } else {
                for (int i = start_idx; i < end_idx; i++) {
                    local_sum += __ldg(input + base_offset + i);
                }
            }
        } else if constexpr (std::is_same<scalar_t, double>::value) {
            if ((end_idx - start_idx) > 0 && ((end_idx - start_idx) % 2 == 0) &&
                (((uintptr_t)(input + base_offset + start_idx) & 0xF) == 0)) {
                int num_vec = (end_idx - start_idx) / 2;
                for (int i = 0; i < num_vec; i++) {
                    double2 val = __ldg(reinterpret_cast<const double2*>(input + base_offset + start_idx) + i);
                    local_sum += val.x + val.y;
                }
            } else {
                for (int i = start_idx; i < end_idx; i++) {
                    local_sum += __ldg(input + base_offset + i);
                }
            }
        } else {
            for (int i = start_idx; i < end_idx; i++) {
                local_sum += __ldg(input + base_offset + i);
            }
        }
    } else { // non-contiguous reduction
        for (int i = start_idx; i < end_idx; i++) {
            local_sum += __ldg(input + base_offset + i * stride);
        }
    }

    // Shared memory reduction within the block
    __shared__ scalar_t sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // Write out the mean for this output element
        output[out_idx] = sdata[0] / static_cast<scalar_t>(L);
    }
}

// Combined Mean Reduction Kernel - Atomic Version
// When L is large, the reduction is split into chunks processed by multiple blocks per output element.
// Each block computes a partial sum over a chunk and uses atomicAdd to accumulate the result.

template <typename scalar_t>
__global__ void combined_mean_reduce_atomic(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,  // intermediate accumulation; pre-initialized to 0
    int L,
    int stride
) {
    // Each output element is processed by a 2D grid: blockIdx.x indexes the output element, blockIdx.y indexes the chunk
    int out_idx = blockIdx.x;
    // Decode out_idx into outer and inner indices
    int outer_idx = out_idx / stride;
    int inner_idx = out_idx % stride;
    int base_offset = outer_idx * (L * stride) + inner_idx;

    // Define the chunk this block will handle
    int chunk_size = BLOCK_SIZE * ITEMS_PER_THREAD;
    int chunk_start = blockIdx.y * chunk_size;
    if (chunk_start >= L) return;
    int current_chunk = (chunk_start + chunk_size < L) ? chunk_size : (L - chunk_start);

    scalar_t local_sum = static_cast<scalar_t>(0);

    // Tiled loop within the current chunk
    int tile_size = (current_chunk + blockDim.x - 1) / blockDim.x;
    int start_idx = threadIdx.x * tile_size;
    int end_idx = (start_idx + tile_size < current_chunk) ? (start_idx + tile_size) : current_chunk;

    if (stride == 1) { // contiguous
        if constexpr (std::is_same<scalar_t, float>::value) {
            if ((end_idx - start_idx) > 0 && ((end_idx - start_idx) % 4 == 0) &&
                (((uintptr_t)(input + base_offset + chunk_start + start_idx) & 0xF) == 0)) {
                int num_vec = (end_idx - start_idx) / 4;
                for (int i = 0; i < num_vec; i++) {
                    float4 val = __ldg(reinterpret_cast<const float4*>(input + base_offset + chunk_start + start_idx) + i);
                    local_sum += val.x + val.y + val.z + val.w;
                }
            } else {
                for (int i = start_idx; i < end_idx; i++) {
                    local_sum += __ldg(input + base_offset + chunk_start + i);
                }
            }
        } else if constexpr (std::is_same<scalar_t, double>::value) {
            if ((end_idx - start_idx) > 0 && ((end_idx - start_idx) % 2 == 0) &&
                (((uintptr_t)(input + base_offset + chunk_start + start_idx) & 0xF) == 0)) {
                int num_vec = (end_idx - start_idx) / 2;
                for (int i = 0; i < num_vec; i++) {
                    double2 val = __ldg(reinterpret_cast<const double2*>(input + base_offset + chunk_start + start_idx) + i);
                    local_sum += val.x + val.y;
                }
            } else {
                for (int i = start_idx; i < end_idx; i++) {
                    local_sum += __ldg(input + base_offset + chunk_start + i);
                }
            }
        } else {
            for (int i = start_idx; i < end_idx; i++) {
                local_sum += __ldg(input + base_offset + chunk_start + i);
            }
        }
    } else { // non-contiguous
        for (int i = start_idx; i < end_idx; i++) {
            local_sum += __ldg(input + base_offset + (chunk_start + i) * stride);
        }
    }

    // Reduction in shared memory
    __shared__ scalar_t sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&output[out_idx], sdata[0]);
    }
}

// Final kernel to finish the mean computation by dividing each accumulated sum by L
template <typename scalar_t>
__global__ void final_divide_kernel(
    scalar_t* __restrict__ output,
    int L,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = output[idx] / static_cast<scalar_t>(L);
    }
}

// Host function: mean_reduce_cuda
// This function selects the appropriate kernel based on the reduction dimension length

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimension
    if (dim < 0) dim += input.dim();

    // Get the input sizes
    auto sizes = input.sizes().vec();
    int64_t L = sizes[dim];

    // Compute outer_size and inner_size
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= sizes[i];
    }
    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < sizes.size(); i++) {
        inner_size *= sizes[i];
    }

    // Total number of output elements
    int64_t N = outer_size * inner_size;
    int stride = inner_size;  // reduction elements are spaced by inner_size

    torch::Tensor output;

    // Decide kernel based on reduction dimension length
    int chunk_size = BLOCK_SIZE * ITEMS_PER_THREAD;
    if (L <= chunk_size) {
        // Use no-atomic reduction: one block per output element
        output = torch::empty({N}, input.options());
        int blocks = N;
        int threads = BLOCK_SIZE;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "combined_mean_reduce_noatomic", ([&] {
            combined_mean_reduce_noatomic<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<int>(L),
                stride,
                static_cast<int>(N)
            );
        }));
    } else {
        // Use atomic reduction: multiple blocks per output element
        output = torch::zeros({N}, input.options());
        int numChunks = (L + chunk_size - 1) / chunk_size;
        dim3 grid(static_cast<int>(N), numChunks);
        int threads = BLOCK_SIZE;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "combined_mean_reduce_atomic", ([&] {
            combined_mean_reduce_atomic<scalar_t><<<grid, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                static_cast<int>(L),
                stride
            );
        }));

        // Finalize the mean by dividing each output by L
        int final_threads = 256;
        int final_blocks = (N + final_threads - 1) / final_threads;
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "final_divide_kernel", ([&] {
            final_divide_kernel<scalar_t><<<final_blocks, final_threads>>>(
                output.data_ptr<scalar_t>(),
                static_cast<int>(L),
                static_cast<int>(N)
            );
        }));
    }

    // Reshape the output to remove the reduced dimension
    sizes.erase(sizes.begin() + dim);
    output = output.view(sizes);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Combined Mean Reduction (CUDA)");
}
