#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void tanh_kernel_gridstride(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int N) {
    
    // Grid-stride loop pattern
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += gridDim.x * blockDim.x) {
        
        // Load input value
        scalar_t val = input[idx];
        
        // Compute tanh and store result
        output[idx] = tanhf(val);
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int N = input.numel();
    
    // Optimize thread and block configuration
    const int threadsPerBlock = 256;
    const int maxBlocks = 65535;
    
    // Calculate optimal number of blocks based on SM count
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    
    // Aim for at least 2 blocks per SM for better occupancy
    const int numSMs = props.multiProcessorCount;
    const int blocksPerSM = 2;
    const int minBlocks = numSMs * blocksPerSM;
    
    // Choose number of blocks based on both data size and hardware capabilities
    const int numBlocks = min(maxBlocks, 
                            max(minBlocks, 
                                (N + threadsPerBlock - 1) / threadsPerBlock));
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "tanh_kernel_gridstride", ([&] {
        tanh_kernel_gridstride<scalar_t><<<numBlocks, threadsPerBlock>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tanh forward with grid stride (CUDA)");
}