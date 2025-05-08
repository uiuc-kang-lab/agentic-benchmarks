#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-level reduction using __shfl_down_sync to compute the maximum in the warp
template <typename scalar_t>
__device__ inline scalar_t warp_reduce_max(scalar_t val) {
    // Use full warp mask
    for (int offset = 16; offset > 0; offset /= 2) {
        scalar_t other = __shfl_down_sync(0xffffffff, val, offset);
        val = (val > other) ? val : other;
    }
    return val;
}

// CUDA kernel for ReLU activation that uses warp-level primitives
// Each thread computes its output, and then a warp-level reduction determines if the entire warp
// has only non-positive inputs (thus all outputs are zero). In that case, only one thread writes zeros
// for the whole warp, reducing redundant global memory writes.

template <typename scalar_t>
__global__ void relu_kernel_warp_shfl(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        // Compute ReLU for the current element
        scalar_t x = input[gid];
        scalar_t y = (x > scalar_t(0)) ? x : scalar_t(0);
        
        // Determine lane within the warp
        int lane = threadIdx.x & 31;
        // Compute the global base index for the warp
        int warp_base = gid - lane;
        // Check if this warp is fully within bounds
        bool full_warp = (warp_base + 32 <= size);
        
        // Use warp-level reduction to find the maximum value in the warp
        // If the maximum is zero, then all y values in the warp are zero
        scalar_t warp_max = warp_reduce_max(y);
        
        if (full_warp && (warp_max == scalar_t(0))) {
            // All outputs in this warp are zero. Use thread 0 to write zeros for the entire warp.
            if (lane == 0) {
                #pragma unroll
                for (int i = 0; i < 32; i++) {
                    output[warp_base + i] = scalar_t(0);
                }
            }
        } else {
            // Otherwise, each thread writes its computed ReLU value.
            output[gid] = y;
        }
    }
}

// PyTorch wrapper using AT_DISPATCH for floating point types
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_warp_shfl", ([&] {
        relu_kernel_warp_shfl<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward using warp-level shfl reduction (CUDA)");
}
