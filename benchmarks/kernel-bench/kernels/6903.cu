#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <vector>

// Device helper: select the better pair (max value, index) in a uniform (branchless) way.
// In case of a tie, the pair with the lower index is chosen.
__device__ __forceinline__ float2 select_max(float2 a, float2 b) {
    // Ternary operator typically maps to a predicated instruction on modern GPUs,
    // ensuring uniform control flow within the warp.
    return (b.x > a.x || (b.x == a.x && b.y < a.y)) ? b : a;
}

// Kernel that computes argmax over a specified dimension using warp-shuffle reduction
// to minimize warp divergence by keeping control flow uniform.
// Each block processes a single (outer, inner) pair corresponding to the input tensor.
__global__ void warp_shuffle_argmax_kernel(
    const float* __restrict__ x,
    int64_t* __restrict__ indices,
    const int outerSize,
    const int dimSize,
    const int innerSize) {

    // Each block processes one output element
    const int idx = blockIdx.x;
    if (idx >= outerSize * innerSize) return;

    // Determine the (outer, inner) pair that this block handles
    const int outer_idx = idx / innerSize;
    const int inner_idx = idx % innerSize;
    const int base_offset = outer_idx * dimSize * innerSize + inner_idx;

    // Each thread computes a local maximum over its assigned subset of the 'dim' dimension
    float local_max = -FLT_MAX;
    int local_index = 0;
    for (int d = threadIdx.x; d < dimSize; d += blockDim.x) {
        float val = x[base_offset + d * innerSize];
        // Regular conditional update; note that divergence here is unavoidable as threads
        // process different data, but is minimized by the subsequent uniform warp reduction.
        if (val > local_max || (val == local_max && d < local_index)) {
            local_max = val;
            local_index = d;
        }
    }

    // Pack the result into a float2 where:
    // .x holds the maximum value and .y encodes the index (via __int_as_float).
    float2 local_pair = make_float2(local_max, __int_as_float(local_index));

    // Warp-level reduction using __shfl_down_sync to combine results in a uniform control flow.
    const unsigned int full_mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        float2 other = __shfl_down_sync(full_mask, local_pair, offset);
        local_pair = select_max(local_pair, other);
    }

    // Store each warp's result into shared memory
    __shared__ float2 warp_results[32];  // Support up to 32 warps per block
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;  // Equivalent to threadIdx.x / 32
    if (lane == 0) {
        warp_results[warp_id] = local_pair;
    }
    __syncthreads();

    // Final reduction among warp-level results, performed by the first warp
    float2 final_val;
    if (threadIdx.x < (blockDim.x >> 5)) {
        final_val = warp_results[threadIdx.x];
    } else {
        final_val = make_float2(-FLT_MAX, __int_as_float(0));
    }
    if (threadIdx.x < 32) {
        for (int offset = 16; offset > 0; offset /= 2) {
            float2 temp = __shfl_down_sync(full_mask, final_val, offset);
            final_val = select_max(final_val, temp);
        }
    }

    // Thread 0 of the block writes the final argmax index to the output
    if (threadIdx.x == 0) {
        indices[idx] = __float_as_int(final_val.y);
    }
}

// Host function to launch the kernel
torch::Tensor argmax_forward_cuda(const torch::Tensor& x, const int64_t dim) {
    TORCH_CHECK(x.scalar_type() == at::kFloat, "Only float32 is supported.");
    
    // Ensure the input tensor is contiguous and get its dimensions
    auto x_contig = x.contiguous();
    auto sizes = x_contig.sizes();
    const int ndim = x_contig.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "Invalid dimension for argmax.");

    // Compute outerSize, dimSize, and innerSize from the tensor's shape
    int outerSize = 1, innerSize = 1;
    const int dimSize = sizes[dim];
    for (int i = 0; i < dim; ++i) {
        outerSize *= sizes[i];
    }
    for (int i = dim + 1; i < ndim; ++i) {
        innerSize *= sizes[i];
    }

    // Construct output shape by removing the 'dim' dimension
    std::vector<int64_t> out_sizes;
    for (int i = 0; i < ndim; ++i) {
        if (i == dim) continue;
        out_sizes.push_back(sizes[i]);
    }
    auto options = torch::TensorOptions().device(x.device()).dtype(torch::kLong);
    auto indices = torch::empty(out_sizes, options);

    // Setup kernel launch parameters: one block per (outer, inner) pair
    const int blocks = outerSize * innerSize;
    const int threads = 256;

    warp_shuffle_argmax_kernel<<<blocks, threads>>>(
        x_contig.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        outerSize,
        dimSize,
        innerSize
    );

    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmax_forward_cuda, "ArgMax CUDA forward (warp shuffle minimal divergence)");
}
