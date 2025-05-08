#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Define an inline device function for tanh that handles both float and double precision.
template <typename scalar_t>
__device__ inline scalar_t custom_tanh(scalar_t a) {
    return tanh(a);  // For double precision
}

// Specialization for float to ensure we use tanhf for float precision
template <>
__device__ inline float custom_tanh<float>(float a) {
    return tanhf(a);
}

// CUDA kernel using a grid-stride loop with precomputed iteration count to reduce warp divergence
template <typename scalar_t>
__global__ void tanh_kernel_grid_stride(const scalar_t* __restrict__ input,
                                          scalar_t* __restrict__ output,
                                          const int size) {
    // Compute the global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute the stride: total number of threads in the grid
    int stride = blockDim.x * gridDim.x;

    // Only threads with a valid starting index execute the loop
    if (tid < size) {
        // Precompute the number of iterations for this thread to avoid checking the condition inside the loop
        int n_iter = ((size - tid - 1) / stride) + 1;
        for (int i = 0; i < n_iter; i++) {
            int index = tid + i * stride;
            output[index] = custom_tanh<scalar_t>(input[index]);
        }
    }
}

// Forward function wrapping the kernel launch

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);

    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_grid_stride", ([&] {
        tanh_kernel_grid_stride<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with grid-stride loop to minimize warp divergence (CUDA)");
}
