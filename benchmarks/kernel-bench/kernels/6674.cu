#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

// This kernel computes the max reduction along the specified dimension using shared memory for intra-block reduction
// and warp-level primitives (__shfl_down_sync) for the final reduction steps.
// Each block computes one output element corresponding to a unique combination of the outer and inner indices.

template <typename scalar_t>
__global__ void shared_warp_max_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t outer_size,
    const int64_t dim_size,
    const int64_t inner_size
) {
    // Each block computes one output element (one reduction over dim_size)
    // Compute the overall output index from blockIdx.x
    int out_index = blockIdx.x;  
    int outer_idx = out_index / inner_size;
    int inner_idx = out_index % inner_size;
    
    // Compute pointer base for this reduction: all elements along the reduction dimension
    const int64_t base = outer_idx * dim_size * inner_size + inner_idx;

    int tid = threadIdx.x;
    
    // Initialize local maximum to the lowest possible value
    // Using std::numeric_limits to get the lowest value for the type
    scalar_t local_max = -std::numeric_limits<scalar_t>::infinity();

    // Stride through the reduction dimension. Each thread processes multiple elements spaced by blockDim.x
    for (int i = tid; i < dim_size; i += blockDim.x) {
        scalar_t val = input[base + i * inner_size];
        local_max = max(local_max, val);
    }
    
    // Allocate shared memory dynamically
    extern __shared__ char shared[];
    scalar_t* sdata = reinterpret_cast<scalar_t*>(shared);
    sdata[tid] = local_max;
    __syncthreads();

    // Intra-block reduction using shared memory (reduce in powers of 2)
    for (unsigned int s = blockDim.x / 2; s > 32; s /= 2) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // For the final warp, use warp-level primitive for efficient reduction
    if (tid < 32) {
        // Use a local variable to hold the value from shared memory
        scalar_t val = sdata[tid];
        // Full mask for active threads in the warp
        unsigned int mask = 0xffffffff;
        val = max(val, __shfl_down_sync(mask, val, 16, 32));
        val = max(val, __shfl_down_sync(mask, val, 8));
        val = max(val, __shfl_down_sync(mask, val, 4));
        val = max(val, __shfl_down_sync(mask, val, 2));
        val = max(val, __shfl_down_sync(mask, val, 1));
        if (tid == 0) {
            output[out_index] = val;
        }
    }
}


// Host function: sets up the grid dimensions and launches the kernel
torch::Tensor max_reduce_cuda_forward(torch::Tensor input, int64_t dim) {
    // Handle negative dimension
    if (dim < 0) {
        dim += input.dim();
    }

    // Compute outer_size: product of dimensions before the reduction dimension
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= input.size(i);
    }

    // Compute inner_size: product of dimensions after the reduction dimension
    int64_t inner_size = 1;
    for (int i = dim + 1; i < input.dim(); i++) {
        inner_size *= input.size(i);
    }

    const int64_t dim_size = input.size(dim);

    // Build the output tensor (input shape with the reduction dimension removed)
    auto output_sizes = input.sizes().vec();
    output_sizes.erase(output_sizes.begin() + dim);
    auto output = torch::empty(output_sizes, input.options());

    // Each output element corresponds to one unique (outer, inner) index
    int total_outputs = outer_size * inner_size;

    // Choose the block size for the reduction along the reduction dimension
    const int threads = 256;
    dim3 grid(total_outputs);

    // Shared memory size for each block: number of threads times the size of one element
    size_t shared_mem = threads * input.element_size();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_reduce_forward", ([&] {
        shared_warp_max_reduce_kernel<scalar_t><<<grid, threads, shared_mem>>>(
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
    m.def("forward", &max_reduce_cuda_forward, "Shared Memory & Warp Optimized Max Reduction Forward (CUDA)");
}
