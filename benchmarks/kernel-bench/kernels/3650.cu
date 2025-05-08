#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Helper function to broadcast a value from a given lane in a warp
template <typename scalar_t>
__device__ inline scalar_t warp_broadcast(scalar_t val, int srcLane) {
    unsigned mask = 0xffffffff;
    if constexpr (std::is_same<scalar_t, float>::value) {
        return __shfl_sync(mask, val, srcLane);
    } else if constexpr (std::is_same<scalar_t, double>::value) {
        // For double, reinterpret as 64-bit integer
        long long int tmp = __double_as_longlong(val);
        tmp = __shfl_sync(mask, tmp, srcLane);
        return __longlong_as_double(tmp);
    }
}

// Clamp function for both float and double types
template <typename scalar_t>
__device__ inline scalar_t clamp_val(scalar_t x) {
    if constexpr (std::is_same<scalar_t, float>::value) {
        return fminf(fmaxf(x, 0.f), 1.f);
    } else {
        return fmin(fmax(x, 0.0), 1.0);
    }
}

// CUDA kernel that computes HardSigmoid using warp-level primitives
// for broadcasting constant values instead of reading them from shared memory
// y = clamp((x + 3) / 6, 0, 1)

template <typename scalar_t>
__global__ void warp_primitive_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                                   scalar_t* __restrict__ output,
                                                   size_t numel) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Each warp broadcasts constant values from lane 0
    scalar_t local_offset = static_cast<scalar_t>(3);
    scalar_t local_scale = static_cast<scalar_t>(1) / static_cast<scalar_t>(6);
    
    // Use warp-level broadcast to retrieve constants
    scalar_t offset = warp_broadcast<scalar_t>(local_offset, 0);
    scalar_t scale = warp_broadcast<scalar_t>(local_scale, 0);
    
    for (size_t i = idx; i < numel; i += stride) {
        scalar_t x = input[i];
        scalar_t y = (x + offset) * scale;
        y = clamp_val(y);
        output[i] = y;
    }
}

// Host function to launch the kernel

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "warp_primitive_hardsigmoid_cuda", ([&] {
        warp_primitive_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA) using warp-level primitives");
}
