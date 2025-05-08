#include <torch/extension.h>
#include <vector>
#include <cfloat>

// Optimized kernel using __ldg for read-only access and aligned memory ops
__global__ void ldg_aligned_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int dimSize,
    const int innerSize) {

    // Each block handles one output element
    int global_idx = blockIdx.x;
    int outer_idx = global_idx / innerSize;
    int inner_idx = global_idx % innerSize;
    int base_offset = outer_idx * dimSize * innerSize + inner_idx;

    extern __shared__ float shared[];
    int* sidx = (int*)&shared[blockDim.x];

    float thread_max = -FLT_MAX;
    int thread_max_idx = 0;

    // Use __ldg for read-only cached access and aligned loads
    for (int i = threadIdx.x; i < dimSize; i += blockDim.x) {
        // 128-bit aligned access using __ldg intrinsic
        float val = __ldg(&x[base_offset + i * innerSize]);
        if (val > thread_max) {
            thread_max = val;
            thread_max_idx = i;
        }
    }

    shared[threadIdx.x] = thread_max;
    sidx[threadIdx.x] = thread_max_idx;
    __syncthreads();

    // Parallel reduction in shared memory
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

torch::Tensor ldg_aligned_argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim.");

    int outerSize = 1;
    for (int d = 0; d < dim; d++) outerSize *= sizes[d];
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) innerSize *= sizes[d];

    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) if (d != dim) out_sizes.push_back(sizes[d]);
    auto indices = torch::empty(out_sizes, torch::TensorOptions().device(x.device()).dtype(torch::kLong));

    // Use larger block size to maximize memory throughput
    int blockSize = 256;
    dim3 grid(outerSize * innerSize);
    size_t shared_mem = blockSize * (sizeof(float) + sizeof(int));

    ldg_aligned_argmax_kernel<<<grid, blockSize, shared_mem>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &ldg_aligned_argmax_forward_cuda, "LDG-aligned ArgMax CUDA forward");
}