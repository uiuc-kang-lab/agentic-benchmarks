#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int THREADS = 512;
const int ELEMENTS_PER_THREAD = 4;

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                             scalar_t* __restrict__ output,
                             const int64_t size) {
    const int global_idx = blockIdx.x * THREADS * ELEMENTS_PER_THREAD + threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        const int idx = global_idx + i * THREADS;
        if (idx < size) {
            float val = static_cast<float>(input[idx]);
            float exp_val = expf(-val);
            output[idx] = static_cast<scalar_t>(1.0f / (1.0f + exp_val));
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    const int blocks = (size + THREADS * ELEMENTS_PER_THREAD - 1) / (THREADS * ELEMENTS_PER_THREAD);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        const auto* input_data = input.data_ptr<scalar_t>();
        auto* output_data = output.data_ptr<scalar_t>();
        
        sigmoid_kernel<scalar_t><<<blocks, THREADS>>>(input_data, output_data, size);
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA) with coalesced memory access");
}