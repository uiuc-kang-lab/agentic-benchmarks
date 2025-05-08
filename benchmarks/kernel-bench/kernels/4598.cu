#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Kernel with dynamic parallelism for even workload distribution

template <typename scalar_t>
__global__ void rms_norm_kernel_dynamic(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int numel,
    const int num_features,
    const float eps
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < numel; i += stride) {
        int batch_id = i / (numel / num_features);
        int offset = i % (numel / num_features);
        int start_idx = batch_id * num_features * (numel / num_features) + offset;

        // Calculate sum of squares
        scalar_t sumsq = 0.0f;
        for (int feat = 0; feat < num_features; feat++) {
            scalar_t val = input[start_idx + feat * (numel / num_features)];
            sumsq += val * val;
        }

        // Calculate RMS
        scalar_t rms = sqrt(sumsq / num_features + eps);

        // Normalize
        for (int feat = 0; feat < num_features; feat++) {
            int idx = start_idx + feat * (numel / num_features);
            output[idx] = input[idx] / rms;
        }
    }
}

// CUDA forward function with dynamic parallelism

torch::Tensor rms_norm_cuda_forward_dynamic(torch::Tensor input, float eps) {
    auto output = torch::empty_like(input);
    
    const int batch_size = input.size(0);
    const int num_features = input.size(1);
    
    // Calculate total number of elements
    int numel = input.numel();

    const int threads_per_block = 256;
    const int blocks = (numel + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "rms_norm_cuda_dynamic", ([&] {
        rms_norm_kernel_dynamic<scalar_t><<<blocks, threads_per_block>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            numel,
            num_features,
            eps
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &rms_norm_cuda_forward_dynamic, "RMS normalization forward with dynamic parallelism (CUDA)");
}