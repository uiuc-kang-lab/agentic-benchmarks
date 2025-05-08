#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// This kernel manually unrolls critical loops to reduce overhead and improve performance.

template <typename scalar_t, int VEC_SIZE>
__global__ void hardsigmoid_kernel_unrolled(const scalar_t* __restrict__ input,
                                             scalar_t* __restrict__ output,
                                             size_t numel) {
    // Each chunk holds VEC_SIZE elements
    size_t num_chunks = numel / VEC_SIZE;
    // Use consecutive thread indices for coalesced memory access
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t total_threads = blockDim.x * gridDim.x;
    
    // Pre-compute constants
    constexpr scalar_t three = static_cast<scalar_t>(3.0);
    constexpr scalar_t sixth = static_cast<scalar_t>(1.0/6.0);
    
    // Select vector type based on precision: for float use float4, for double use double2
    using vec_t = typename std::conditional<
        std::is_same<scalar_t, float>::value,
        float4,
        double2
    >::type;
    
    // Ensure memory alignment
    static_assert(sizeof(vec_t) % sizeof(scalar_t) == 0, "Vector size must be aligned");

    // Evenly process full vectorized chunks
    for (size_t idx = tid; idx < num_chunks; idx += total_threads) {
        size_t base = idx * VEC_SIZE;
        vec_t chunk = *reinterpret_cast<const vec_t*>(&input[base]);
        scalar_t elems[VEC_SIZE];
        *reinterpret_cast<vec_t*>(elems) = chunk;

        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            scalar_t x = elems[i];
            x = (x + three) * sixth;  // computes (x + 3) / 6
            x = (x < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) :
                (x > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : x);
            elems[i] = x;
        }

        *reinterpret_cast<vec_t*>(&output[base]) = *reinterpret_cast<vec_t*>(elems);
    }

    // Process any remaining tail elements
    size_t tail_start = num_chunks * VEC_SIZE;
    for (size_t i = tail_start + tid; i < numel; i += total_threads) {
        scalar_t x = input[i];
        x = (x + three) * sixth;
        x = (x < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) :
            (x > static_cast<scalar_t>(1) ? static_cast<scalar_t>(1) : x);
        output[i] = x;
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_unrolled_optimized_cuda", ([&] {
        // Choose vector size depending on type: 4 for float (float4), 2 for double (double2)
        constexpr int vec_size = std::is_same<scalar_t, float>::value ? 4 : 2;
        size_t num_chunks = numel / vec_size;
        int blocks = (num_chunks + threads - 1) / threads;
        if (blocks == 0) {
            blocks = 1;
        }
        hardsigmoid_kernel_unrolled<scalar_t, vec_size><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel);
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "HardSigmoid activation forward with unrolled loops (CUDA)");
}
