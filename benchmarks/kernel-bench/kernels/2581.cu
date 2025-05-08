#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
}

// CUDA kernel for ReLU activation using vectorized loads/stores
template <typename scalar_t>
__global__ void relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    // Grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < size; 
         idx += blockDim.x * gridDim.x) {
        
        scalar_t val = input[idx];
        output[idx] = val > 0 ? val : 0;
    }
}

// Specialized kernel for float4 processing
__global__ void relu_kernel_vec4(
    float4* __restrict__ output,
    const float4* __restrict__ input,
    const int64_t size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_size = size / 4;
    
    for (int i = idx; i < vec_size; i += stride) {
        float4 val = input[i];
        val.x = val.x > 0 ? val.x : 0;
        val.y = val.y > 0 ? val.y : 0;
        val.z = val.z > 0 ? val.z : 0;
        val.w = val.w > 0 ? val.w : 0;
        output[i] = val;
    }
    
    // Handle remaining elements
    if (idx == 0) {
        for (int i = vec_size * 4; i < size; i++) {
            float val = reinterpret_cast<const float*>(input)[i];
            reinterpret_cast<float*>(output)[i] = val > 0 ? val : 0;
        }
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;  // Keep block size at 256 (good for occupancy)
    const int max_blocks = 65535;
    const int blocks = std::min(max_blocks, (int)((input.numel() + threads - 1) / threads));

    // Create CUDA stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel", ([&] {
        if (std::is_same<scalar_t, float>::value && (input.numel() >= 1024)) {
            // Use vectorized version for float type with large inputs
            relu_kernel_vec4<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<float4*>(output.data_ptr<scalar_t>()),
                reinterpret_cast<const float4*>(input.data_ptr<scalar_t>()),
                input.numel()
            );
        } else {
            // Use regular version for other types or small inputs
            relu_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                input.numel()
            );
        }
    }));

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Synchronize stream
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward (CUDA)");
}