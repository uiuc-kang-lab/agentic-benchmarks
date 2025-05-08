#include <torch/extension.h>
#include <vector>

// CUDA kernel for argmax using shared memory to reduce global memory access latency.
__global__ void argmax_shared_memory_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize)
{
    extern __shared__ float shared_data[];

    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Total number of (outer, inner) pairs
    int total = outerSize * innerSize;

    if (idx < total) {
        // Identify which outer index and inner index this thread corresponds to
        int outer_idx = idx / innerSize;
        int inner_idx = idx % innerSize;

        // The input offset for the start of the dimension slice
        int start_offset = (outer_idx * dimSize * innerSize) + inner_idx;

        // Load data into shared memory
        for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
            shared_data[d] = x[start_offset + d * innerSize];
        }
        __syncthreads();

        float max_val = shared_data[0];
        int max_idx = 0;

        // Step over the dimension
        for (int d = 1; d < dimSize; d++) {
            float val = shared_data[d];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }

        // Write the max index for this pair
        indices[outer_idx * innerSize + inner_idx] = max_idx;
    }
}

// Host function to launch the CUDA kernel
torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    // Ensure input tensor is float32 (can adapt as needed)
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");

    // We will use contiguous data
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    auto ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute sizes for outerSize, dimSize, innerSize
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // The output shape is the input shape with dimension dim removed
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }

    // Create output for indices (type: long)
    auto options = torch::TensorOptions()
                       .device(x.device())
                       .dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Launch kernel
    const int threads = 256;
    const int total = outerSize * innerSize;
    const int blocks = (total + threads - 1) / threads;

    // Allocate shared memory size
    size_t shared_mem_size = dimSize * sizeof(float);

    argmax_shared_memory_kernel<<<blocks, threads, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward");
}