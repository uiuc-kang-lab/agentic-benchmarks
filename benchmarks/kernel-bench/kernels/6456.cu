#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <type_traits>

// Define block configuration
#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4


// Kernel for single-block reduction (no atomic), used when the reduction dimension is small
// Each block processes one output element
template <typename scalar_t>
__global__ void mean_reduce_kernel_noatomic(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int L,           // reduction dimension length
    int stride,      // stride for accessing reduction elements
    int N            // number of output elements
) {
    int out_idx = blockIdx.x;  // each block handles one output element (flattened)
    if (out_idx >= N) return;
    
    // Decode the flat output index into outer and inner indices
    int outer_idx = out_idx / stride;
    int inner_idx = out_idx % stride;
    // Base offset for the current output element's reduction slice in the input
    int base_offset = outer_idx * (L * stride) + inner_idx;

    __shared__ scalar_t sdata[BLOCK_SIZE];
    scalar_t sum = static_cast<scalar_t>(0);

    // Each thread accumulates over a strided loop
    for (int i = threadIdx.x; i < L; i += blockDim.x) {
         sum += __ldg(input + base_offset + i * stride);
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
         if (threadIdx.x < s)
              sdata[threadIdx.x] += sdata[threadIdx.x + s];
         __syncthreads();
    }
    
    if (threadIdx.x == 0) {
         // Write the final mean directly (no atomic needed)
         output[out_idx] = sdata[0] / static_cast<scalar_t>(L);
    }
}

// Kernel for multi-block reduction using atomic adds.
// When the reduction dimension L is large, we split the work across multiple blocks per output element.
// Each block computes a partial sum for a slice of the reduction dimension and atomically accumulates into global memory.
template <typename scalar_t>
__global__ void mean_reduce_kernel_atomic(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,  // intermediate accumulation; should be pre-initialized to 0
    int L,
    int stride
) {
    // out_idx identifies the output element this block contributes to
    int out_idx = blockIdx.x; 

    // Decode the flat output index into outer and inner indices
    int outer_idx = out_idx / stride;
    int inner_idx = out_idx % stride;
    int base_offset = outer_idx * (L * stride) + inner_idx;

    // Each block processes a contiguous chunk along the reduction dimension
    int chunk_start = blockIdx.y * (BLOCK_SIZE * ITEMS_PER_THREAD);
    
    __shared__ scalar_t sdata[BLOCK_SIZE];
    scalar_t local_sum = static_cast<scalar_t>(0);
    
    int total = L;
    // Each thread processes ITEMS_PER_THREAD elements in a strided manner
    for (int i = threadIdx.x; i < BLOCK_SIZE * ITEMS_PER_THREAD; i += blockDim.x) {
         int idx = chunk_start + i;
         if (idx < total) {
              local_sum += __ldg(input + base_offset + idx * stride);
         }
    }
    
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Reduce partial sums in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
         if (threadIdx.x < s)
              sdata[threadIdx.x] += sdata[threadIdx.x + s];
         __syncthreads();
    }

    if (threadIdx.x == 0) {
         // Atomically accumulate the block's partial sum to the global output
         atomicAdd(&output[out_idx], sdata[0]);
    }
}

// Final kernel to divide the accumulated sums by L to get the mean
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

// Host function that selects the appropriate kernel launch based on reduction dimension size
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    if (dim < 0) dim += input.dim();

    // Get input sizes and compute L, outer_size, and inner_size
    std::vector<int64_t> sizes = input.sizes().vec();
    int64_t L = sizes[dim];

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
    int stride = inner_size;  // For computing input index: reduction elements are spaced by inner_size

    torch::Tensor output;
    
    // Determine if we need multi-block reduction
    int chunk_size = BLOCK_SIZE * ITEMS_PER_THREAD;
    if (L <= chunk_size) {
         // Use single-block reduction per output element; no atomic operations needed.
         output = torch::empty({N}, input.options());
         int blocks = N;
         int threads = BLOCK_SIZE;
         AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda_noatomic", ([&] {
              mean_reduce_kernel_noatomic<scalar_t><<<blocks, threads>>>(
                 input.data_ptr<scalar_t>(),
                 output.data_ptr<scalar_t>(),
                 static_cast<int>(L),
                 stride,
                 static_cast<int>(N)
              );
         }));
    } else {
         // Use multi-block reduction with atomics to reduce contention
         // Allocate intermediate output and initialize to 0
         output = torch::zeros({N}, input.options());
         int numChunks = (L + chunk_size - 1) / chunk_size;
         dim3 grid(static_cast<int>(N), numChunks);
         int threads = BLOCK_SIZE;
         AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda_atomic", ([&] {
              mean_reduce_kernel_atomic<scalar_t><<<grid, threads>>>(
                 input.data_ptr<scalar_t>(),
                 output.data_ptr<scalar_t>(),
                 static_cast<int>(L),
                 stride
              );
         }));
         // Finalize the mean by dividing by L
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

    // Reshape output to remove the reduced dimension
    sizes.erase(sizes.begin() + dim);
    output = output.view(sizes);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Mean Reduction (Atomic Optimized CUDA)");
}
