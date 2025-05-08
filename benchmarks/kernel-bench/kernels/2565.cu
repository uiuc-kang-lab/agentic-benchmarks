#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel for ReLU activation
// Combines vectorization and block size tuning
template <typename scalar_t>
__global__ void optimized_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Vectorized processing for aligned data
    if constexpr (sizeof(scalar_t) == 4) {
        constexpr int VEC_SIZE = 4;
        using vec_t = float4;
        const int vec_size = size / VEC_SIZE;
        const vec_t* in_vec = reinterpret_cast<const vec_t*>(input);
        vec_t* out_vec = reinterpret_cast<vec_t*>(output);

        for (int i = idx; i < vec_size; i += stride) {
            vec_t val = __ldg(&in_vec[i]);
            val.x = val.x > 0 ? val.x : 0;
            val.y = val.y > 0 ? val.y : 0;
            val.z = val.z > 0 ? val.z : 0;
            val.w = val.w > 0 ? val.w : 0;
            out_vec[i] = val;
        }

        // Handle remaining elements
        const int scalar_idx = vec_size * VEC_SIZE + idx;
        if (scalar_idx < size) {
            for (int i = scalar_idx; i < size; i += stride) {
                output[i] = __ldg(&input[i]) > 0 ? input[i] : 0;
            }
        }
    } else {
        for (int i = idx; i < size; i += stride) {
            output[i] = __ldg(&input[i]) > 0 ? input[i] : 0;
        }
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    // Select block size based on input size
    int threads;
    if (size > 1048576) threads = 512;
    else if (size > 10240) threads = 256;
    else threads = 128;

    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_relu_kernel", ([&] {
        optimized_relu_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized ReLU forward (CUDA)");
}