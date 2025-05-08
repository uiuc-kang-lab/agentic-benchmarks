#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int BLOCK_SIZE>
__global__ void optimized_relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size,
    const bool use_vectorized) {

    const int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int stride = gridDim.x * BLOCK_SIZE;

    if (use_vectorized && sizeof(scalar_t) == 4) {
        // Vectorized processing using float4
        constexpr int VEC_SIZE = 4;
        using vec_t = float4;
        const int vec_size = size / VEC_SIZE;
        const vec_t* in_vec = reinterpret_cast<const vec_t*>(input);
        vec_t* out_vec = reinterpret_cast<vec_t*>(output);

        #pragma unroll 2
        for (int i = idx; i < vec_size; i += stride) {
            vec_t val = __ldg(&in_vec[i]);
            val.x = fmaxf(val.x, 0.0f);
            val.y = fmaxf(val.y, 0.0f);
            val.z = fmaxf(val.z, 0.0f);
            val.w = fmaxf(val.w, 0.0f);
            out_vec[i] = val;
        }

        // Handle remaining elements
        const int scalar_idx = vec_size * VEC_SIZE + idx;
        if (scalar_idx < size) {
            for (int i = scalar_idx; i < size; i += stride) {
                output[i] = fmaxf(__ldg(&input[i]), 0.0f);
            }
        }
    } else {
        // Non-vectorized processing with memory coalescing
        #pragma unroll 4
        for (int i = idx; i < size; i += stride) {
            output[i] = fmaxf(__ldg(&input[i]), 0.0f);
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();

    // Adaptive block size selection
    int threads;
    if (size > 1048576) threads = 512;
    else if (size > 10240) threads = 256;
    else threads = 128;

    const int blocks = min(65535, (size + threads - 1) / threads);
    const bool use_vectorized = (size >= 1024) && (input.stride(0) == 1);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "optimized_relu_kernel", ([&] {
        switch(threads) {
            case 512:
                optimized_relu_kernel<scalar_t, 512><<<blocks, 512>>>(
                    output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), 
                    size, use_vectorized);
                break;
            case 256:
                optimized_relu_kernel<scalar_t, 256><<<blocks, 256>>>(
                    output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), 
                    size, use_vectorized);
                break;
            default:
                optimized_relu_kernel<scalar_t, 128><<<blocks, 128>>>(
                    output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), 
                    size, use_vectorized);
        }
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Adaptive ReLU forward (CUDA)");
}