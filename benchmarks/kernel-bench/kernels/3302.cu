#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_THREAD 4

__global__ void coalesced_swish_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int64_t n
) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_threads = blockDim.x * gridDim.x;
    const int thread_id = bid * blockDim.x + tid;
    
    for (int i = thread_id * ELEMENTS_PER_THREAD; 
         i < n; 
         i += num_threads * ELEMENTS_PER_THREAD) {
        
        float4 in_val;
        float4 out_val;
        
        if (i + ELEMENTS_PER_THREAD <= n) {
            in_val = *reinterpret_cast<const float4*>(input + i);
            
            #pragma unroll
            for (int j = 0; j < ELEMENTS_PER_THREAD; j++) {
                float val = ((float*)&in_val)[j];
                float sigmoid = __fdividef(1.0f, (1.0f + __expf(-val)));
                ((float*)&out_val)[j] = val * sigmoid;
            }
            
            *reinterpret_cast<float4*>(output + i) = out_val;
        } else {
            for (int j = 0; i + j < n; j++) {
                float val = input[i + j];
                float sigmoid = __fdividef(1.0f, (1.0f + __expf(-val)));
                output[i + j] = val * sigmoid;
            }
        }
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    const int blocks = std::min(
        (int)((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
        1024
    );
    
    coalesced_swish_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Coalesced memory access Swish forward (CUDA)");
}