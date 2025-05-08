#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Device function to compute HardSigmoid: y = clamp((x + 3) / 6, 0, 1)
template <typename scalar_t>
__device__ inline scalar_t hardsigmoid_func(scalar_t x) {
    scalar_t y = (x + static_cast<scalar_t>(3)) / static_cast<scalar_t>(6);
    return (y < static_cast<scalar_t>(0)) ? static_cast<scalar_t>(0) : 
           (y > static_cast<scalar_t>(1)) ? static_cast<scalar_t>(1) : y;
}

// Vectorized kernel using pack types to ensure memory coalescing
// For float, we use float4 (pack_size = 4); for double, we use double2 (pack_size = 2).

template <typename scalar_t, typename pack_t, int pack_size>
__global__ void vectorized_hardsigmoid_kernel(const scalar_t* __restrict__ input,
                                              scalar_t* __restrict__ output,
                                              size_t numel) {
    // Number of complete packs
    size_t num_pack = numel / pack_size;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Reinterpret memory as vectorized packs
    const pack_t* input_pack = reinterpret_cast<const pack_t*>(input);
    pack_t* output_pack = reinterpret_cast<pack_t*>(output);

    // Process vectorized elements
    for (size_t i = idx; i < num_pack; i += stride) {
        pack_t in_pack = input_pack[i];
        pack_t out_pack;
        // Process each element in the pack
        scalar_t* in_vals = reinterpret_cast<scalar_t*>(&in_pack);
        scalar_t* out_vals = reinterpret_cast<scalar_t*>(&out_pack);
        #pragma unroll
        for (int j = 0; j < pack_size; j++) {
            out_vals[j] = hardsigmoid_func<scalar_t>(in_vals[j]);
        }
        output_pack[i] = out_pack;
    }

    // Handle leftover elements that don't fit in a complete pack
    size_t remainder_start = num_pack * pack_size;
    for (size_t i = remainder_start + idx; i < numel; i += stride) {
        scalar_t x = input[i];
        output[i] = hardsigmoid_func<scalar_t>(x);
    }
}

// Forward function dispatching the appropriate kernel

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA");
    auto output = torch::empty_like(input);
    size_t numel = input.numel();

    int threads = 1024;
    int blocks = 0;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "vectorized_coalesced_hardsigmoid_cuda", ([&] {
        if (std::is_same<scalar_t, float>::value) {
            constexpr int pack_size = 4;
            using pack_t = float4;
            // Adjust blocks for the vectorized loop
            blocks = ((numel / pack_size) + threads - 1) / threads;
            vectorized_hardsigmoid_kernel<scalar_t, pack_t, pack_size><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                numel);
        } else if (std::is_same<scalar_t, double>::value) {
            constexpr int pack_size = 2;
            using pack_t = double2;
            blocks = ((numel / pack_size) + threads - 1) / threads;
            vectorized_hardsigmoid_kernel<scalar_t, pack_t, pack_size><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                numel);
        }
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "Vectorized, coalesced HardSigmoid activation forward (CUDA)");
}
