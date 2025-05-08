#include <torch/extension.h>
#include <vector>
#include <cfloat>

__global__ void divergence_free_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize) {

    int global_idx = blockIdx.x;
    int outer_idx = global_idx / innerSize;
    int inner_idx = global_idx % innerSize;
    int base_offset = outer_idx * dimSize * innerSize + inner_idx;

    extern __shared__ float shared[];
    int* sidx = (int*)&shared[blockDim.x];

    // Initialize with first valid value
    float thread_max = -FLT_MAX;
    int thread_max_idx = 0;
    
    // Main loop - use predicated assignments instead of branches
    #pragma unroll 4
    for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
        float val = __ldg(&x[base_offset + i * innerSize]);
        // Use arithmetic comparison to avoid branching
        bool is_greater = val > thread_max;
        thread_max = is_greater ? val : thread_max;
        thread_max_idx = is_greater ? i : thread_max_idx;
    }

    // Store in shared memory
    shared[threadIdx.x] = thread_max;
    sidx[threadIdx.x] = thread_max_idx;
    __syncthreads();

    // Reduction using warp-level operations for the first 32 threads
    if (threadIdx.x < 32) {
        // Warp-level reduction using shuffle operations
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xffffffff, thread_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, thread_max_idx, offset);
            
            // Predicated assignment instead of branching
            bool is_greater = other_val > thread_max;
            thread_max = is_greater ? other_val : thread_max;
            thread_max_idx = is_greater ? other_idx : thread_max_idx;
        }
    }

    // First thread writes the final result
    if (threadIdx.x == 0) {
        indices[global_idx] = thread_max_idx;
    }
}

torch::Tensor divergence_free_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 supported");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim");

    // Compute sizes
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Prepare output tensor
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d != dim) out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Launch kernel with optimized parameters
    const int blockSize = 128;  // Multiple of warp size for efficient reduction
    dim3 grid(outerSize * innerSize);
    dim3 block(blockSize);
    
    size_t shared_mem_size = blockSize * (sizeof(float) + sizeof(int));

    divergence_free_argmax_kernel<<<grid, block, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &divergence_free_argmax_forward_cuda, "Divergence-free ArgMax CUDA forward");
}