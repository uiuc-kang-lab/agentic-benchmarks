#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int THREADS = 256;
const int ELEMENTS_PER_THREAD = 4;
const int SHARED_MEM_SIZE = THREADS * ELEMENTS_PER_THREAD;

// Kernel optimized by reviewing thread and block indexing

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                             scalar_t* __restrict__ output,
                             const int64_t size) {
    __shared__ float shared_data[SHARED_MEM_SIZE];
    
    const int tid = threadIdx.x;
    const int global_idx = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + tid;
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = global_idx + i * blockDim.x;
        if (idx < size) {
            shared_data[tid + i * blockDim.x] = static_cast<float>(input[idx]);
        }
    }
    __syncthreads();
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = global_idx + i * blockDim.x;
        if (idx < size) {
            float val = -shared_data[tid + i * blockDim.x];
            float exp_val = __expf(val);
            float r = __fdividef(1.0f, (1.0f + exp_val));
            output[idx] = static_cast<scalar_t>(r);
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
    m.def("forward", &forward, "Optimized Sigmoid forward (CUDA)");
}
