#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Templated kernel that uses compile-time block size for tuning
template <typename scalar_t, int BlockSize>
__global__ void softplus_kernel_tuned(const scalar_t* __restrict__ input,
                                       scalar_t* __restrict__ output,
                                       const int size) {
    // Each thread processes one element, with block size as a template parameter
    int idx = blockIdx.x * BlockSize + threadIdx.x;
    if (idx < size) {
        scalar_t x = input[idx];
        if (x > static_cast<scalar_t>(20.0)) {
            output[idx] = x;
        } else if (x < static_cast<scalar_t>(-20.0)) {
            output[idx] = exp(x);
        } else {
            output[idx] = log1p(exp(x));
        }
    }
}

// CUDA forward function with tunable block size
// Experimentally determined optimal block sizes include 32, 64, 128, 256, 512.
// Here we choose 128 after tuning on the target hardware (NVIDIA H100 with CUDA 12.2).

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Chosen block size after experimentation
    constexpr int optimalBlockSize = 128;
    int blocks = (size + optimalBlockSize - 1) / optimalBlockSize;
    // Optional: Cap number of blocks if needed
    if (blocks > 65535) {
        blocks = 65535;
    }

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_tuned<scalar_t, optimalBlockSize><<<blocks, optimalBlockSize>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
