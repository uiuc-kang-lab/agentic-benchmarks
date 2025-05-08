#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <type_traits>

// Device function that computes HardSigmoid using FMA and branchless clamping
// HardSigmoid: y = clamp((x + 3) / 6, 0, 1) == x/6 + 0.5, clamped
// Using FMA to fuse operations and inline if constexpr to select proper intrinsics

template <typename scalar_t>
__device__ inline scalar_t fast_hardsigmoid(scalar_t x) {
    if constexpr (std::is_same<scalar_t, float>::value) {
        // Use FMA: (x/6 + 0.5) and branchless clamp via fminf/fmaxf
        scalar_t y = __fmaf_rn(x, 1.f / 6.f, 0.5f);
        return fminf(fmaxf(y, 0.f), 1.f);
    } else {
        // For double precision
        scalar_t y = fma(x, static_cast<scalar_t>(1) / static_cast<scalar_t>(6), static_cast<scalar_t>(0.5));
        return fmin(fmax(y, static_cast<scalar_t>(0)), static_cast<scalar_t>(1));
    }
}

// Vectorized kernel processing 4 elements per iteration to reduce loop overhead
// and ensure uniform control flow across threads

template <typename scalar_t>
__global__ void vectorized_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                                 scalar_t* __restrict__ output,
                                                 size_t numel) {
    // Calculate thread index and overall stride
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    // Process groups of 4 elements for better memory throughput
    size_t num_vec = numel / 4;  // Number of complete groups
    for (size_t i = idx; i < num_vec; i += stride) {
        size_t base = i * 4;
        scalar_t x0 = input[base];
        scalar_t x1 = input[base + 1];
        scalar_t x2 = input[base + 2];
        scalar_t x3 = input[base + 3];
        output[base]     = fast_hardsigmoid(x0);
        output[base + 1] = fast_hardsigmoid(x1);
        output[base + 2] = fast_hardsigmoid(x2);
        output[base + 3] = fast_hardsigmoid(x3);
    }
    
    // Process any remaining elements that don't fit into a group of 4
    size_t remainder_start = num_vec * 4;
    for (size_t i = remainder_start + idx; i < numel; i += stride) {
        output[i] = fast_hardsigmoid(input[i]);
    }
}

// Host function to launch the vectorized kernel

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    // Use 1024 threads per block
    const int threads = 1024;
    // Compute number of vectorized groups of 4 elements
    size_t num_vec = numel / 4;
    int blocks = (num_vec > 0) ? static_cast<int>((num_vec + threads - 1) / threads) : 1;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "vectorized_minwarp_hardsigmoid_cuda", ([&] {
        vectorized_hardsigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized HardSigmoid activation forward (CUDA) minimizing warp divergence");
}
