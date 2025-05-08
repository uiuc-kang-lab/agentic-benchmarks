#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tanh_vectorized_opt_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_size = (size + 3) / 4;

    // Process full vectorized elements
    for (int i = tid; i < vec_size; i += stride) {
        float4 in = reinterpret_cast<const float4*>(input)[i];
        float4 out;
        out.x = tanhf(in.x);
        out.y = tanhf(in.y);
        out.z = tanhf(in.z);
        out.w = tanhf(in.w);
        reinterpret_cast<float4*>(output)[i] = out;
    }

    // Process remaining elements (0-3)
    const int rem_start = vec_size * 4;
    const int rem_tid = tid;
    if (rem_tid < size - rem_start) {
        output[rem_start + rem_tid] = tanhf(input[rem_start + rem_tid]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int numel = input.numel();
    
    const int threads = 512;
    const int vec_size = numel / 4;
    const int blocks = (vec_size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.type(), "tanh_vectorized_opt_kernel", ([&] {
        tanh_vectorized_opt_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Vectorized Tanh forward (CUDA)");
}