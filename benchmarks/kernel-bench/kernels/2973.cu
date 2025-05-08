#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// Device function that applies tanh in full precision
// Uses tanhf for float and tanh for double
template <typename scalar_t>
__device__ inline scalar_t device_tanh(scalar_t x) {
    if constexpr (std::is_same<scalar_t, float>::value) {
        return tanhf(x);
    } else {
        return tanh(x);
    }
}

// Kernel that applies tanh elementwise using a grid-stride loop
// No shared memory is used and threads work independently,
// so no __syncthreads() is necessary, which avoids excessive synchronizations.
template <typename scalar_t>
__global__ void tanh_nosync_kernel(const scalar_t* __restrict__ input,
                                      scalar_t* __restrict__ output,
                                      const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements by striding
    for (int i = idx; i < size; i += stride) {
        output[i] = device_tanh<scalar_t>(input[i]);
    }
}

// Host function that launches the kernel
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_nosync_kernel", ([&] {
        tanh_nosync_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "No synchronization optimized Tanh forward (CUDA)");
}
