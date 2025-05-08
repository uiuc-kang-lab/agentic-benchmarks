#include <torch/extension.h>
#include <vector>
#include <cfloat>

// Device function to compute the local maximum for each thread
__device__ void compute_local_max(const float* __restrict__ x, int base_offset, int dimSize, int innerSize, float &local_max, int &local_max_idx) {
    for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
        float val = x[base_offset + i * innerSize];
        if (val > local_max) {
            local_max = val;
            local_max_idx = i;
        }
    }
}

// Device function to perform block-level reduction to get the maximum value and corresponding index
__device__ void block_reduce_argmax(volatile float* s_val, volatile int* s_idx, int block_size) {
    int tid = threadIdx.x;
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_val[tid + s] > s_val[tid]) {
                s_val[tid] = s_val[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }
}

// Kernel: Each block handles one output element ((outer, inner) pair) and reduces over the 'dim' dimension
__global__ void modular_device_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize) {

    // Compute global output index corresponding to an (outer, inner) pair
    int global_idx = blockIdx.x;
    int outer_idx = global_idx / innerSize;
    int inner_idx = global_idx % innerSize;

    // Base offset for this slice in the input tensor
    int base_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Allocate shared memory: first part for max values, second for indices
    extern __shared__ char smem[];
    float* s_val = (float*)smem;
    int* s_idx = (int*)(smem + blockDim.x * sizeof(float));

    // Each thread computes its local maximum
    float thread_max = -FLT_MAX;
    int thread_max_idx = 0;
    compute_local_max(x, base_offset, dimSize, innerSize, thread_max, thread_max_idx);

    // Store thread local results in shared memory
    s_val[threadIdx.x] = thread_max;
    s_idx[threadIdx.x] = thread_max_idx;
    __syncthreads();

    // Perform block-level reduction using a modular device function
    block_reduce_argmax(s_val, s_idx, blockDim.x);

    // The first thread writes the result
    if (threadIdx.x == 0) {
        indices[global_idx] = s_idx[0];
    }
}

// Host function to launch the modular device argmax kernel

torch::Tensor modular_device_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute sizes for outer, dim, and inner dimensions
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // The output shape is the input shape with the 'dim' dimension removed
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    int total_outputs = outerSize * innerSize;
    int blockSize = 128;  // Chosen block size
    dim3 grid(total_outputs);
    dim3 block(blockSize);

    // Shared memory: for max values and indices
    size_t shared_mem_size = blockSize * (sizeof(float) + sizeof(int));

    modular_device_argmax_kernel<<<grid, block, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize
    );

    return indices;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modular_device_argmax_forward_cuda, "Modular Device Function ArgMax CUDA forward");
}
