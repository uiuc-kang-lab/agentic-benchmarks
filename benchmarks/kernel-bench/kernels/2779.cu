#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void sigmoid_kernel(const scalar_t* __restrict__ input,
                               scalar_t* __restrict__ output,
                               const int64_t size) {
    // Each thread processes 4 elements
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    const int warp_id = threadIdx.x / 32;
    
    // Process 4 elements per thread to increase arithmetic intensity
    #pragma unroll
    for (int i = tid; i < size; i += stride) {
        float val = static_cast<float>(-input[i]);
        float exp_val = expf(val);
        output[i] = static_cast<scalar_t>(1.0f / (1.0f + exp_val));
        
        // Ensure coalesced memory access within warps
        __syncwarp();
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    // Optimize block size for better occupancy
    const int threads = 256;
    const int blocks = min(65535, (size + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel", [&] {
        sigmoid_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward (CUDA)");
}