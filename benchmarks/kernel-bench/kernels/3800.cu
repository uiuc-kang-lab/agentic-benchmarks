#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel using warp-level primitives to reduce divergence and improve performance

template <typename scalar_t>
__global__ void softplus_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        scalar_t x = __ldg(&input[idx]);
        
        // Get the active mask for the current warp
        unsigned int mask = __activemask();
        // Lane index within warp
        int lane = threadIdx.x & (warpSize - 1);

        // Initialize flags for branch conditions
        int high_flag = (x > 20.0) ? 1 : 0;
        int low_flag  = (x < -20.0) ? 1 : 0;

        // Use warp-level reduction with __shfl_down_sync to sum the flags across the warp
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            high_flag += __shfl_down_sync(mask, high_flag, offset);
            low_flag  += __shfl_down_sync(mask, low_flag, offset);
        }

        // Determine the number of active threads in this warp
        int active_count = __popc(mask);
        // Broadcast the reduced sums from lane 0 to all lanes
        int warp_high = __shfl_sync(mask, high_flag, 0);
        int warp_low  = __shfl_sync(mask, low_flag, 0);
        
        // If all threads in the warp satisfy the condition, pick the fast approximation
        if (warp_high == active_count) {
            // For large x, softplus(x) ≈ x
            output[idx] = x;
        } else if (warp_low == active_count) {
            // For very negative x, softplus(x) ≈ exp(x)
            output[idx] = exp(x);
        } else {
            // Mixed warp: compute softplus in a numerically stable manner
            if (x > 0) {
                output[idx] = x + log1p(exp(-x));
            } else {
                output[idx] = log1p(exp(x));
            }
        }
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;  // 8 warps per block
    const int blocks = (size + threads - 1) / threads;
    
    // Ensure proper L2 cache alignment
    cudaFuncSetAttribute(
        softplus_kernel<scalar_t>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxL1);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
