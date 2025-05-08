#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Constants defining the tile size (number of output elements processed per block) and the number of threads for reduction per output element
#define TILE 8
#define REDUCE_THREADS 32

// Kernel that distributes the reduction work evenly across a 2D thread block.
// Each block processes TILE output elements. The x-dimension indexes which output element in the tile,
// and the y-dimension partitions the work for the reduction along the reduction dimension L.

template <typename scalar_t>
__global__ void even_workload_mean_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int L,       // length of reduction dimension
    int stride,  // stride (inner_size) to traverse the reduction dimension
    int N        // total number of output elements
) {
    // Allocate shared memory dynamically; size: TILE * REDUCE_THREADS elements
    extern __shared__ char smem[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(smem);

    // Indices in the 2D block
    int tile_idx = threadIdx.x;      // which output element in the tile (0 to TILE-1)
    int reduce_idx = threadIdx.y;      // thread's index for reduction work (0 to REDUCE_THREADS-1)

    // Compute global output index
    int global_output_idx = blockIdx.x * TILE + tile_idx;
    if (global_output_idx >= N) return;

    // Decode the global output index into (outer, inner) indices
    // Input shape: [outer_size, L, inner_size]
    // Here, stride = inner_size
    int outer_idx = global_output_idx / stride;
    int inner_idx = global_output_idx % stride;
    int base_offset = outer_idx * (L * stride) + inner_idx;

    // Each thread accumulates a partial sum over the reduction dimension using a grid-stride loop
    scalar_t sum = static_cast<scalar_t>(0);
    for (int i = reduce_idx; i < L; i += REDUCE_THREADS) {
         sum += __ldg(input + base_offset + i * stride);
    }

    // Store the partial sum into shared memory
    int shmem_idx = tile_idx * REDUCE_THREADS + reduce_idx;
    sdata[shmem_idx] = sum;
    __syncthreads();

    // Perform reduction along the y-dimension for each output element in the tile
    for (int s = REDUCE_THREADS / 2; s > 0; s >>= 1) {
        if (reduce_idx < s) {
            sdata[shmem_idx] += sdata[shmem_idx + s];
        }
        __syncthreads();
    }

    // The first thread in the y-dimension writes the final result (mean) to global memory
    if (reduce_idx == 0) {
         output[global_output_idx] = sdata[tile_idx * REDUCE_THREADS] / static_cast<scalar_t>(L);
    }
}

// Host function to setup and launch the kernel
torch::Tensor mean_reduce_cuda(torch::Tensor input, int64_t dim) {
    // Handle negative dimensions
    if (dim < 0) dim += input.dim();

    // Get input sizes and compute L (length along reduction dimension), outer_size, and inner_size
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
    
    // Total number of output elements (after reducing the dimension)
    int64_t N = outer_size * inner_size;
    int stride = inner_size;  // stride to jump across the reduction dimension in input

    // Create a 1D output tensor; later we will reshape it
    auto output = torch::empty({N}, input.options());

    // Determine grid and block dimensions
    // Each block processes TILE output elements
    int grid_x = (N + TILE - 1) / TILE;
    dim3 grid(grid_x);
    dim3 block(TILE, REDUCE_THREADS);

    // Shared memory size in bytes: TILE * REDUCE_THREADS * sizeof(scalar_t)
    size_t shared_mem_size = TILE * REDUCE_THREADS * sizeof(float);  // placeholder, will be set correctly in dispatch below

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "mean_reduce_cuda", ([&] {
         shared_mem_size = TILE * REDUCE_THREADS * sizeof(scalar_t);
         even_workload_mean_reduce_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
             input.data_ptr<scalar_t>(),
             output.data_ptr<scalar_t>(),
             static_cast<int>(L),
             stride,
             static_cast<int>(N)
         );
    }));

    // Reshape the output to remove the reduced dimension
    sizes.erase(sizes.begin() + dim);
    output = output.view(sizes);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mean_reduce_cuda, "Even Workload Mean Reduction (CUDA)");
}
