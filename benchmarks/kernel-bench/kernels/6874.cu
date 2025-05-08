#include <torch/extension.h>
#include <vector>
#include <float.h>

// This kernel computes the argmax over a specified dimension.
// It uses __ldg() to optimize read-only global memory loads and assumes that the input is 128-bit aligned.
// Each block handles one slice (an outer-inner pair) and performs a cooperative reduction in shared memory using warp-level operations.

__global__ void argmax_kernel_ldg(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    // Each block processes one slice (one outer-inner pair)
    int slice = blockIdx.x;
    if (slice >= outerSize * innerSize) return;

    int outer_idx = slice / innerSize;
    int inner_idx = slice % innerSize;
    int base_offset = outer_idx * (dimSize * innerSize) + inner_idx;

    // Each thread computes a local maximum for a subset of the reduction dimension
    float local_max = -FLT_MAX;
    int local_argmax = 0;

    // Use __ldg() to load data from global memory (read-only cache), assuming proper alignment
    for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
        float curr_val = __ldg(&x[base_offset + d * innerSize]);
        if (__ldg(&x[base_offset + d * innerSize]) > local_max) {
            local_max = curr_val;
            local_argmax = d;
        }
    }

    // Shared memory for reduction across threads in a block
    extern __shared__ char shared_mem[];
    float* s_max = reinterpret_cast<float*>(shared_mem);
    int* s_idx = reinterpret_cast<int*>(s_max + blockDim.x);

    s_max[threadIdx.x] = local_max;
    s_idx[threadIdx.x] = local_argmax;
    __syncthreads();

    // Reduction in shared memory: standard tree reduction
    for (unsigned int stride = blockDim.x / 2; stride > 64; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_max[threadIdx.x + stride] > s_max[threadIdx.x]) {
                s_max[threadIdx.x] = s_max[threadIdx.x + stride];
                s_idx[threadIdx.x] = s_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    // Warp-level reduction without __syncthreads()
    if (threadIdx.x < 64) {
        if (s_max[threadIdx.x + 64] > s_max[threadIdx.x]) {
            s_max[threadIdx.x] = s_max[threadIdx.x + 64];
            s_idx[threadIdx.x] = s_idx[threadIdx.x + 64];
        }
        if (s_max[threadIdx.x + 32] > s_max[threadIdx.x]) {
            s_max[threadIdx.x] = s_max[threadIdx.x + 32];
            s_idx[threadIdx.x] = s_idx[threadIdx.x + 32];
        }
        if (s_max[threadIdx.x + 16] > s_max[threadIdx.x]) {
            s_max[threadIdx.x] = s_max[threadIdx.x + 16];
            s_idx[threadIdx.x] = s_idx[threadIdx.x + 16];
        }
        if (s_max[threadIdx.x + 8] > s_max[threadIdx.x]) {
            s_max[threadIdx.x] = s_max[threadIdx.x + 8];
            s_idx[threadIdx.x] = s_idx[threadIdx.x + 8];
        }
        if (s_max[threadIdx.x + 4] > s_max[threadIdx.x]) {
            s_max[threadIdx.x] = s_max[threadIdx.x + 4];
            s_idx[threadIdx.x] = s_idx[threadIdx.x + 4];
        }
        if (s_max[threadIdx.x + 2] > s_max[threadIdx.x]) {
            s_max[threadIdx.x] = s_max[threadIdx.x + 2];
            s_idx[threadIdx.x] = s_idx[threadIdx.x + 2];
        }
        if (s_max[threadIdx.x + 1] > s_max[threadIdx.x]) {
            s_max[threadIdx.x] = s_max[threadIdx.x + 1];
            s_idx[threadIdx.x] = s_idx[threadIdx.x + 1];
        }
    }

    // Write the result
    if (threadIdx.x == 0) {
        indices[slice] = s_idx[0];
    }
}

// Host function to launch the CUDA kernel

torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();

    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    // Compute outerSize (product of dims before 'dim'), dimSize, and innerSize (product of dims after 'dim')
    int outerSize = 1;
    for (int d = 0; d < dim; d++) {
        outerSize *= sizes[d];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int d = dim + 1; d < ndim; d++) {
        innerSize *= sizes[d];
    }

    // Prepare output shape: remove the reduced dimension
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d == dim) continue;
        out_sizes.push_back(sizes[d]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Each slice corresponds to one (outer, inner) pair
    int slices = outerSize * innerSize;

    // Use 128 threads per block for good occupancy; shared memory holds 128 floats and 128 ints
    const int threads = 128;
    int blocks = slices;
    int sharedMemSize = threads * (sizeof(float) + sizeof(int));

    argmax_kernel_ldg<<<blocks, threads, sharedMemSize>>>(
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
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward using __ldg() for optimized global memory loads");
}
