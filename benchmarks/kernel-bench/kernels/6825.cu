#include <torch/extension.h>
#include <vector>
#include <cfloat>

// Kernel: Each block handles one output element and reduces over the 'dim' dimension
// using shared memory. This version combines the best practices from both kernels.
__global__ void optimized_argmax_kernel(
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

    float thread_max = -FLT_MAX;
    int thread_max_idx = 0;

    for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
        float val = x[base_offset + i * innerSize];
        if (val > thread_max) {
            thread_max = val;
            thread_max_idx = i;
        }
    }

    shared[threadIdx.x] = thread_max;
    sidx[threadIdx.x] = thread_max_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared[threadIdx.x + s] > shared[threadIdx.x]) {
                shared[threadIdx.x] = shared[threadIdx.x + s];
                sidx[threadIdx.x] = sidx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        indices[global_idx] = sidx[0];
    }
}

// Host function to launch the optimized argmax kernel

torch::Tensor optimized_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
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
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    int total_outputs = outerSize * innerSize;
    int blockSize = 128;
    dim3 grid(total_outputs);
    dim3 block(blockSize);

    size_t shared_mem_size = blockSize * (sizeof(float) + sizeof(int));

    optimized_argmax_kernel<<<grid, block, shared_mem_size>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize
    );

    return indices;
}

// Pybind11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_argmax_forward_cuda, "Optimized ArgMax CUDA forward");
}