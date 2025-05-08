#include <torch/extension.h>
#include <vector>
#include <cfloat>

// Device function: compute local maximum using __ldg for memory efficiency
__device__ __forceinline__ void compute_local_max(const float* __restrict__ x, int base_offset, int dimSize, int innerSize, float &local_max, int &local_max_idx) {
    // Each thread computes its local maximum over the assigned slice
    for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
        // Use __ldg to fetch data from global memory with read-only cache
        float val = __ldg(&x[base_offset + i * innerSize]);
        if (val > local_max) {
            local_max = val;
            local_max_idx = i;
        }
    }
}

// Combined ArgMax kernel: each block computes the argmax for one (outer, inner) slice
__global__ void combined_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize) {

    // Map block index to (outer, inner) coordinates
    int global_idx = blockIdx.x;
    int outer_idx = global_idx / innerSize;
    int inner_idx = global_idx % innerSize;
    int base_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Allocate shared memory dynamically: first half for float values and second half for indices
    extern __shared__ float shared[];
    int* s_idx = (int*) &shared[blockDim.x];

    // Each thread finds its local maximum over the 'dim' dimension
    float thread_max = -FLT_MAX;
    int thread_max_idx = 0;
    compute_local_max(x, base_offset, dimSize, innerSize, thread_max, thread_max_idx);

    // Store the local maximum and index in shared memory
    shared[threadIdx.x] = thread_max;
    s_idx[threadIdx.x] = thread_max_idx;
    __syncthreads();

    // Perform reduction in shared memory to find the block-wide maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shared[threadIdx.x + stride] > shared[threadIdx.x]) {
                shared[threadIdx.x] = shared[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes the result for this (outer, inner) slice
    if (threadIdx.x == 0) {
        indices[global_idx] = s_idx[0];
    }
}

// Host function to launch the combined argmax kernel
torch::Tensor combined_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dimension for argmax.");

    // Compute the sizes for outer, dim, and inner dimensions
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Construct the output shape (input shape with the 'dim' dimension removed)
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    int total_outputs = outerSize * innerSize;
    int blockSize = 128;  // Optimized block size determined through tuning
    dim3 grid(total_outputs);
    dim3 block(blockSize);

    // Shared memory allocation: blockSize floats for values + blockSize ints for indices
    size_t shared_mem_size = blockSize * (sizeof(float) + sizeof(int));

    combined_argmax_kernel<<<grid, block, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &combined_argmax_forward_cuda, "Combined ArgMax CUDA forward");
}
