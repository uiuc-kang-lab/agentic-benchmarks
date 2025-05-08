#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

// Define block configuration
#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4

// Stage 1 Kernel: Compute partial sums over chunks of the reduction dimension without using atomics
// Grid dimensions: gridDim.x = N (total output elements), gridDim.y = num_chunks (chunks per output element)

template <typename scalar_t>
__global__ void partial_mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ intermediate,
    int L,       // Length of the reduction dimension
    int stride,  // Stride between consecutive elements along the reduction dimension (equals inner_size)
    int num_chunks // Number of chunks to split L
) {
    // Each block handles a specific output element and a specific chunk
    int out_idx = blockIdx.x;  // Unique output element index (flattened)
    int chunk = blockIdx.y;    // Which chunk of the reduction dimension is processed by this block

    int chunk_size = BLOCK_SIZE * ITEMS_PER_THREAD;
    int start = chunk * chunk_size;
    int end = start + chunk_size;
    if (end > L) end = L;

    // Decode out_idx into outer and inner indices
    int outer_idx = out_idx / stride;
    int inner_idx = out_idx % stride;
    int base_offset = outer_idx * (L * stride) + inner_idx;

    __shared__ scalar_t sdata[BLOCK_SIZE];
    scalar_t sum = static_cast<scalar_t>(0);

    // Each thread sums a subset of elements in this chunk
    for (int i = threadIdx.x; i < (end - start); i += blockDim.x) {
         int idx = start + i;  // Index within the reduction dimension
         // Safety check (idx < L) is redundant due to loop bound but kept for clarity
         if (idx < L) {
             sum += __ldg(input + base_offset + idx * stride);
         }
    }

    sdata[threadIdx.x] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
         if (threadIdx.x < s) {
              sdata[threadIdx.x] += sdata[threadIdx.x + s];
         }
         __syncthreads();
    }

    // Write the block's partial sum into the intermediate buffer
    if (threadIdx.x == 0) {
         intermediate[out_idx * num_chunks + chunk] = sdata[0];
    }
}

// Stage 2 Kernel: Sum the partial sums from each output element and divide by L to compute the mean

template <typename scalar_t>
__global__ void final_mean_reduce_kernel(
    const scalar_t* __restrict__ intermediate,
    scalar_t* __restrict__ output,
    int num_chunks,
    int L,
    int N  // Total number of output elements
) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx < N) {
         scalar_t sum = static_cast<scalar_t>(0);
         for (int i = 0; i < num_chunks; i++) {
              sum += intermediate[out_idx * num_chunks + i];
         }
         output[out_idx] = sum / static_cast<scalar_t>(L);
    }
}

// Host function setting up the two-stage kernel

torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Adjust negative dimension
    if (dim < 0) dim += input.dim();

    // Get input shape and compute L, outer_size, and inner_size
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

    int64_t N = outer_size * inner_size;  // Total number of output elements

    // Determine chunk parameters to split the reduction dimension
    int chunk_size = BLOCK_SIZE * ITEMS_PER_THREAD;
    int num_chunks = (static_cast<int>(L) + chunk_size - 1) / chunk_size;

    // Allocate an intermediate buffer [N, num_chunks] (flattened) to store partial sums
    auto intermediate = torch::empty({N, num_chunks}, input.options());

    const int threads = BLOCK_SIZE;
    // Grid for stage1: x-dimension corresponds to each output element, y-dimension corresponds to chunks
    dim3 grid1(static_cast<int>(N), num_chunks);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "partial_mean_reduce_kernel", ([&] {
         partial_mean_reduce_kernel<scalar_t><<<grid1, threads>>>(
              input.data_ptr<scalar_t>(),
              intermediate.data_ptr<scalar_t>(),
              static_cast<int>(L),
              static_cast<int>(inner_size),
              num_chunks
         );
    }));

    // Allocate output tensor (1D) to store the final reduction result for each output element
    auto output = torch::empty({N}, input.options());
    const int final_threads = 256;
    int final_blocks = (N + final_threads - 1) / final_threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "final_mean_reduce_kernel", ([&] {
         final_mean_reduce_kernel<scalar_t><<<final_blocks, final_threads>>>(
              intermediate.data_ptr<scalar_t>(),
              output.data_ptr<scalar_t>(),
              num_chunks,
              static_cast<int>(L),
              static_cast<int>(N)
         );
    }));

    // Remove the reduced dimension from the shape and reshape the output tensor
    sizes.erase(sizes.begin() + dim);
    output = output.view(sizes);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Two-Stage Mean Reduction without Excessive Atomics (CUDA)");
}
