#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_kernel_shared(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    // Shared memory buffer - using float4 for vectorized access
    extern __shared__ char shared_mem[];
    float4* shared_data = reinterpret_cast<float4*>(shared_mem);
    
    const int tid = threadIdx.x;
    const int vec_size = 4;
    const int items_per_block = blockDim.x * vec_size;
    int base_idx = blockIdx.x * items_per_block;
    
    // Vector types for efficient memory access
    float4* in_vec = (float4*)input;
    float4* out_vec = (float4*)output;
    
    while (base_idx < size) {
        // Load into shared memory using vectorized loads
        for (int i = 0; i < vec_size; i++) {
            if (base_idx + tid * vec_size + i < size) {
                shared_data[tid] = in_vec[base_idx/vec_size + tid];
            }
        }
        __syncthreads();
        
        // Process data in shared memory
        if (base_idx + tid * vec_size < size) {
            float4 val = shared_data[tid];
            
            // Apply ReLU to vector components
            val.x = val.x > 0 ? val.x : 0;
            val.y = val.y > 0 ? val.y : 0;
            val.z = val.z > 0 ? val.z : 0;
            val.w = val.w > 0 ? val.w : 0;
            
            // Write back to global memory
            out_vec[base_idx/vec_size + tid] = val;
        }
        
        base_idx += gridDim.x * items_per_block;
        __syncthreads();
    }
    
    // Handle remaining elements
    if (blockIdx.x == gridDim.x - 1) {
        const int remaining_start = (size / (vec_size * blockDim.x)) * (vec_size * blockDim.x);
        for (int i = remaining_start + tid; i < size; i += blockDim.x) {
            output[i] = input[i] > 0 ? input[i] : 0;
        }
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = std::min(256, (int)((input.numel() / 4 + threads - 1) / threads));
    const int shared_mem_size = threads * sizeof(float4);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_shared", ([&] {
        relu_kernel_shared<scalar_t><<<blocks, threads, shared_mem_size>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward (CUDA)");
}