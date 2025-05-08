#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

// Device function to select the best (value, index) pair.
// Returns b if its value is larger or equal (with lower index) than a; otherwise returns a.
__device__ inline float2 select_max(const float2 a, const float2 b) {
    if (b.x > a.x || (b.x == a.x && b.y < a.y))
        return b;
    return a;
}

// Device function for warp-level reduction using shuffle intrinsics.
__device__ inline float2 warp_reduce(float2 val) {
    for (int offset = warpSize / 2; __syncwarp(); offset > 0; offset /= 2) {
        float2 other;
        other.x = __shfl_down_sync(0xffffffff, val.x, offset);
        other.y = __shfl_down_sync(0xffffffff, val.y, offset);
        val = select_max(val, other);
    }
    return val;
}

// CUDA kernel: each block processes one (outer, inner) pair and computes argmax over the given dimension.
// The reduction is split into a warp-level reduction (using the modular warp_reduce function) and
// a block-level reduction using shared memory.
__global__ void argmax_modular_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    int k = blockIdx.x;
    if (k >= outerSize * innerSize)
        return;

    int outer_idx = k / innerSize;
    int inner_idx = k % innerSize;
    int start_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Each thread computes the local maximum and its index over a subset of the dimension.
    float thread_max = -INFINITY;
    int thread_arg = -1;
    for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
        float val = x[start_offset + d * innerSize];
        if (val > thread_max || (val == thread_max && d < thread_arg)) {
            thread_max = val;
            thread_arg = d;
        }
    }
    float2 local = make_float2(thread_max, (float)thread_arg);

    // Perform warp-level reduction using modular device function.
    local = warp_reduce(local);

    // Shared memory allocation for block-level reduction (one float2 per warp).
    extern __shared__ __align__(sizeof(float2)) unsigned char shared[];
    float2* sdata = reinterpret_cast<float2*>(shared);

    int warpId = threadIdx.x / warpSize;
    // Only the first thread in each warp writes its result to shared memory.
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        sdata[warpId] = local;
    }
    __syncthreads();

    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    float2 block_result;
    // First numWarps threads load the warp results; others initialize to -INFINITY.
    if (threadIdx.x < numWarps) {
        block_result = sdata[threadIdx.x];
    } else {
        block_result = make_float2(-INFINITY, -1);
    }
    // Final reduction within the first warp.
    if (threadIdx.x < warpSize) {
        block_result = warp_reduce(block_result);
    }
    if (threadIdx.x == 0) {
        indices[k] = (int)block_result.y;
    }
}

// Host function to launch the CUDA kernel.
// Computes argmax over the specified dimension of the input tensor 'x'.
// The result is a tensor with the 'dim' removed.
torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
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

    // Build the output sizes by omitting the specified dimension.
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndim; d++) {
        if (d != dim) {
            out_sizes.push_back(sizes[d]);
        }
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Launch one block per (outer, inner) pair.
    const int threads = 256;
    const int blocks = outerSize * innerSize;
    // Shared memory: one float2 per warp.
    const size_t sharedMem = ((threads + warpSize - 1) / warpSize) * sizeof(float2);

    argmax_modular_kernel<<<blocks, threads, sharedMem>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (modular device functions optimized)");
}
