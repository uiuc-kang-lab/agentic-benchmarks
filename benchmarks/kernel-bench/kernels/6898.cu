#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>

// This kernel computes argmax along a specified dimension using loop unrolling
// to reduce loop overhead in both the input scan and reduction stages.

template <typename T>
__global__ void argmax_unrolled_kernel(
    const T* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    // Allocate shared memory with proper alignment for float2
    extern __shared__ __align__(sizeof(float2)) unsigned char shmem[];
    float2* sdata = reinterpret_cast<float2*>(shmem);

    int k = blockIdx.x;
    if (k >= outerSize * innerSize) return;

    // Identify the (outer, inner) pair this block is working on
    int outer_idx = k / innerSize;
    int inner_idx = k % innerSize;
    int start_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Each thread computes a partial argmax for its assigned subset
    T thread_max = -std::numeric_limits<T>::infinity();
    int thread_idx = -1;

    // Unroll the loop to reduce overhead; each thread covers indices in steps of blockDim.x
    #pragma unroll
    for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
        float val = x[start_offset + d * innerSize];
        if (val > thread_max || (val == thread_max && d < thread_idx)) {
            thread_max = val;
            thread_idx = d;
        }
    }

    // Write the thread's partial result to shared memory
    sdata[threadIdx.x] = make_float2(thread_max, thread_idx);
    __syncthreads();

    // Reduce within the block using unrolled loops
    #pragma unroll
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            float2 other = sdata[threadIdx.x + s];
            if (other.x > sdata[threadIdx.x].x ||
               (other.x == sdata[threadIdx.x].x && other.y < sdata[threadIdx.x].y)) {
                sdata[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    // Warp-level reduction with unrolling (no need for __syncthreads within a warp)
    #pragma unroll
    for (int s = 32; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float2 other = sdata[threadIdx.x + s];
            if (other.x > sdata[threadIdx.x].x ||
                (other.x == sdata[threadIdx.x].x && other.y < sdata[threadIdx.x].y)) {
                sdata[threadIdx.x] = other;
            }
        }
    }

    // The first thread writes the final result
    if (threadIdx.x == 0) {
        indices[k] = static_cast<int64_t>(sdata[0].y);
    }
}

// Host function to launch the argmax kernel
torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dim for argmax.");

    int outerSize = 1, innerSize = 1, dimSize = sizes[dim];
    for (int i = 0; i < dim; i++) {
        outerSize *= sizes[i];
    }
    for (int i = dim + 1; i < ndim; i++) {
        innerSize *= sizes[i];
    }

    // Build output shape by removing the 'dim' dimension
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < ndim; i++) {
        if (i != dim)
            out_sizes.push_back(sizes[i]);
    }
    
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Launch parameters
    const int threads = 256;
    const int blocks = outerSize * innerSize;
    const size_t smem = threads * sizeof(float2);

    argmax_unrolled_kernel<float><<<blocks, threads, smem>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (loop unrolled)");
}
