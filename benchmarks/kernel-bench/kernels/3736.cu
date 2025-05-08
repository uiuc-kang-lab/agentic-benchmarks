#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Store frequently accessed, read-only parameters in constant memory
__constant__ float const_params_f[2] = {3.0f, 6.0f};
__constant__ double const_params_d[2] = {3.0, 6.0};

// This templated kernel uses vectorized loads/stores and constant memory to retrieve the parameters
// for the HardSigmoid activation: y = clamp((x + 3) / 6, 0, 1).
// VEC_SIZE is 4 for float (using float4) and 2 for double (using double2).

template <typename scalar_t, int VEC_SIZE>
__global__ void hardsigmoid_const_vectorized_kernel(const scalar_t* __restrict__ input,
                                                      scalar_t* __restrict__ output,
                                                      size_t numel) {
    // Calculate number of complete vectorized chunks
    size_t num_chunks = numel / VEC_SIZE;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    // Load constant parameters from __constant__ memory
    scalar_t offset, scale;
    if constexpr (std::is_same<scalar_t, float>::value) {
        offset = const_params_f[0];
        scale  = const_params_f[1];
    } else {
        offset = const_params_d[0];
        scale  = const_params_d[1];
    }
    scalar_t inv_scale = static_cast<scalar_t>(1.0) / scale;

    // Define the vectorized type based on precision: float4 for floats, double2 for doubles
    using vec_t = typename std::conditional<
        std::is_same<scalar_t, float>::value,
        float4,
        double2
    >::type;

    // Process complete vectorized chunks
    for (size_t idx = tid; idx < num_chunks; idx += total_threads) {
        size_t base = idx * VEC_SIZE;
        // Vectorized load
        vec_t chunk = *reinterpret_cast<const vec_t*>(&input[base]);
        scalar_t elems[VEC_SIZE];
        *reinterpret_cast<vec_t*>(elems) = chunk;
        
        // Apply HardSigmoid on each element of the vector using fused multiply-add
        #pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
            scalar_t val = elems[i];
            // Use fused multiply-add (FMA) for better performance
            #ifdef __CUDA_ARCH__
                val = fmaf(val, inv_scale, offset * inv_scale);  // Computes x/6 + 3/6 in one FMA operation
            #else
                val = (val + offset) * inv_scale;
            #endif
            // Clamp between 0 and 1 using min/max intrinsics
            val = max(static_cast<scalar_t>(0), min(static_cast<scalar_t>(1), val));
            elems[i] = val;
        }
        // Vectorized store
        *reinterpret_cast<vec_t*>(&output[base]) = *reinterpret_cast<vec_t*>(elems);
    }

    // Process any remaining tail elements not handled by vectorized loads
    size_t tail_start = num_chunks * VEC_SIZE;
    for (size_t i = tail_start + tid; i < numel; i += total_threads) {
        scalar_t val = input[i];
        val = (val + offset) * inv_scale;
        if (val < static_cast<scalar_t>(0))
            val = static_cast<scalar_t>(0);
        else if (val > static_cast<scalar_t>(1))
            val = static_cast<scalar_t>(1);
        output[i] = val;
    }
}


torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();
    const int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "hardsigmoid_const_vectorized_cuda", ([&] {
        // Select vector size: use 4 for float (float4) and 2 for double (double2)
        constexpr int vec_size = std::is_same<scalar_t, float>::value ? 4 : 2;
        size_t num_chunks = numel / vec_size;
        int blocks = (num_chunks + threads - 1) / threads;
        if (blocks == 0) {
            blocks = 1;
        }
        hardsigmoid_const_vectorized_kernel<scalar_t, vec_size><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel
        );
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "HardSigmoid activation forward (CUDA) with constant memory vectorized kernel");
}
