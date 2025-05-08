#include <torch/extension.h>
#include <vector>
#include <cfloat>

#define BLOCK_SIZE 256
#define DIM_THRESHOLD 512

// Grid-stride kernel for small dimensions
__global__ __launch_bounds__(BLOCK_SIZE)
void tuned_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = outerSize * innerSize;

    if (idx < total) {
        int outer_idx = idx / innerSize;
        int inner_idx = idx % innerSize;
        int start_offset = outer_idx * dimSize * innerSize + inner_idx;

        float max_val = x[start_offset];
        int max_idx = 0;

        for (int d = 1; d < dimSize; d++) {
            float val = x[start_offset + d * innerSize];
            if (val > max_val) {
                max_val = val;
                max_idx = d;
            }
        }
        indices[outer_idx * innerSize + inner_idx] = max_idx;
    }
}

// Block-reduce kernel for large dimensions
__global__ void per_output_block_argmax_kernel(
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

torch::Tensor adaptive_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    int outerSize = 1;
    for (int d = 0; d < dim; d++) outerSize *= sizes[d];
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) innerSize *= sizes[d];

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) if(d != dim) out_sizes.push_back(sizes[d]);
    auto indices = torch::empty(out_sizes, torch::TensorOptions().device(x.device()).dtype(torch::kLong));

    if (dimSize <= DIM_THRESHOLD) {
        int total = outerSize * innerSize;
        int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        tuned_argmax_kernel<<<blocks, BLOCK_SIZE>>>(
            x_contig.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            outerSize,
            dimSize,
            innerSize
        );
    } else {
        int total_outputs = outerSize * innerSize;
        int blockSize = 128;
        size_t shared_mem_size = blockSize * (sizeof(float) + sizeof(int));
        per_output_block_argmax_kernel<<<total_outputs, blockSize, shared_mem_size>>>(
            x_contig.data_ptr<float>(),
            indices.data_ptr<int64_t>(),
            dimSize,
            innerSize
        );
    }

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &adaptive_argmax_forward_cuda, "Adaptive Block Strategy ArgMax");
}