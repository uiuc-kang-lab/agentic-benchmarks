#include <torch/extension.h>
#include <vector>

#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4

__global__ void shared_mem_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {
    
    __shared__ float shared_data[BLOCK_SIZE];
    __shared__ int shared_indices[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;

    if (idx < total) {
        int outer_idx = idx / innerSize;
        int inner_idx = idx % innerSize;
        int start_offset = outer_idx * dimSize * innerSize + inner_idx;

        // Initialize with first value
        float max_val = x[start_offset];
        int max_idx = 0;

        // Process multiple items per thread without synchronization
        #pragma unroll
        for (int d = 1; d < dimSize; d++) {
            float val = x[start_offset + d * innerSize];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }

        // Store thread's result in shared memory
        shared_data[threadIdx.x] = max_val;
        shared_indices[threadIdx.x] = max_idx;

        // Only synchronize once after all threads have written their results
        __syncthreads();

        // Write final result
        indices[outer_idx * innerSize + inner_idx] = shared_indices[threadIdx.x];
    }
}

torch::Tensor shared_mem_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    auto ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }

    auto options = torch::TensorOptions()
                       .device(x.device())
                       .dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    const int total = outerSize * innerSize;
    const int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    shared_mem_argmax_kernel<<<blocks, BLOCK_SIZE>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_mem_argmax_forward_cuda, "ArgMax CUDA forward with shared memory");
}