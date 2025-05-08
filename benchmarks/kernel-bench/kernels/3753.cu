#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void softplus_kernel_vector4(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = tid * 4;

    if (idx + 3 < size) {
        // Vectorized load for 4 elements
        const float4 vec_in = reinterpret_cast<const float4*>(input + idx)[0];
        scalar_t results[4] = {vec_in.x, vec_in.y, vec_in.z, vec_in.w};

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const scalar_t x = results[i];
            results[i] = (x > 20.0) ? x : (x < -20.0) ? exp(x) : log1p(exp(x));
        }

        // Vectorized store
        reinterpret_cast<float4*>(output + idx)[0] = {results[0], results[1], results[2], results[3]};
    } else {
        // Handle remaining elements
        for (int i = 0; i < 4; ++i) {
            if (idx + i < size) {
                const scalar_t x = input[idx + i];
                output[idx + i] = (x > 20.0) ? x : (x < -20.0) ? exp(x) : log1p(exp(x));
            }
        }
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    const int threads = 256;
    const int elements_per_block = threads * 4;
    const int blocks = (size + elements_per_block - 1) / elements_per_block;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "softplus_forward_cuda", ([&] {
        softplus_kernel_vector4<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}