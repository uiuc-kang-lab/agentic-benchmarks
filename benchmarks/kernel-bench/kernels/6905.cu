#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

// Constant memory for frequently accessed read-only parameters
__constant__ int c_outerSize;
__constant__ int c_dimSize;
__constant__ int c_innerSize;

// CUDA kernel that uses constant memory for dimension parameters
// to compute the argmax over a specified dimension.
// Each block processes one (outer, inner) pair. Threads in the block work together
// to compute the argmax over the dimension using shared memory reduction.
__global__ void argmax_const_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices) {

    // Total number of (outer, inner) pairs
    int total = c_outerSize * c_innerSize;
    int k = blockIdx.x;
    if (k >= total) return;

    // Determine outer and inner indices
    int outer_idx = k / c_innerSize;
    int inner_idx = k % c_innerSize;
    int start_offset = outer_idx * c_dimSize * c_innerSize + inner_idx;

    // Allocate shared memory for reduction; each element holds (max_value, index)
    extern __shared__ float2 sdata[];

    // Each thread computes a local maximum over a portion of the c_dimSize elements
    float thread_max = -INFINITY;
    int thread_arg = 0;
    
    // Loop over the dimension with a stride equal to blockDim.x via thread index
    for (int d = threadIdx.x; d < c_dimSize; d += blockDim.x) {
        float val = x[start_offset + d];
        if (val > thread_max) {
            thread_max = val;
            thread_arg = d;
        }
    }

    // Store the thread's result in shared memory
    sdata[threadIdx.x] = make_float2(thread_max, static_cast<float>(thread_arg));
    __syncthreads();

    // Perform reduction in shared memory to get the maximum value and corresponding index
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            float2 other = sdata[threadIdx.x + s];
            float2 current = sdata[threadIdx.x];
            if (other.x > current.x || (other.x == current.x && other.y < current.y)) {
                sdata[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        #pragma unroll
        for (int s = 32; s > 0; s >>= 1) {
            float2 other = sdata[threadIdx.x + s];
            float2 current = sdata[threadIdx.x];
            if (other.x > current.x || (other.x == current.x && other.y < current.y)) {
                sdata[threadIdx.x] = other;
            }
        }
    }

    if (threadIdx.x == 0) {
        indices[k] = static_cast<int64_t>(sdata[0].y);
    }
}

// Host function to launch the CUDA kernel
torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dimension for argmax.");

    // Compute outerSize, dimSize, and innerSize
    int outerSize = 1;
    for (int i = 0; i < dim; ++i) {
        outerSize *= sizes[i];
    }
    int dimSize = sizes[dim];
    int innerSize = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        innerSize *= sizes[i];
    }

    // Build output shape by removing the specified dimension
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < ndim; ++i) {
        if (i != dim)
            out_sizes.push_back(sizes[i]);
    }
    auto indices = torch::empty(out_sizes, x.options().dtype(torch::kLong));

    // Copy the dimension parameters to constant memory
    cudaMemcpyToSymbol(c_outerSize, &outerSize, sizeof(int));
    cudaMemcpyToSymbol(c_dimSize, &dimSize, sizeof(int));
    cudaMemcpyToSymbol(c_innerSize, &innerSize, sizeof(int));

    // Launch parameters
    const int threads = 256;
    const int blocks = outerSize * innerSize;
    size_t shared_mem = threads * sizeof(float2);

    argmax_const_kernel<<<blocks, threads, shared_mem>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>()
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (constant memory optimization)");
}
