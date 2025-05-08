#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_kernel_vectorized(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int vec_size = 4;
    
    // Shared memory for tiling
    extern __shared__ char shared_mem[];
    vec_t* shared_in = reinterpret_cast<vec_t*>(shared_mem);
    
    // Vector types for efficient memory access
    using vec_t = float4;
    vec_t* in_vec = (vec_t*)input;
    vec_t* out_vec = (vec_t*)output;
    const int vec_elements = size / vec_size;
    
    // Calculate tile size and number of tiles
    const int tile_size = blockDim.x;  // Number of vec4 elements per tile
    const int num_tiles = (vec_elements + tile_size - 1) / tile_size;
    
    // Process tiles
    for (int tile = 0; tile < num_tiles; tile++) {
        const int idx = tile * tile_size + tid;
        
        // Load tile into shared memory
        if (idx < vec_elements) {
            shared_in[tid] = in_vec[idx];
        }
        __syncthreads();
        
        // Process elements in shared memory
        if (idx < vec_elements) {
            vec_t val = shared_in[tid];
            
            // Apply ReLU to each component
            val.x = val.x > 0 ? val.x : 0;
            val.y = val.y > 0 ? val.y : 0;
            val.z = val.z > 0 ? val.z : 0;
            val.w = val.w > 0 ? val.w : 0;
            
            out_vec[idx] = val;
        }
        __syncthreads();
    }
    
    // Handle remaining elements
    const int remaining_start = vec_elements * vec_size;
    const int global_tid = bid * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    for (int i = remaining_start + global_tid; i < size; i += stride) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = std::min(256, (int)((input.numel() / 4 + threads - 1) / threads));
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_kernel_vectorized", ([&] {
        relu_kernel_vectorized<scalar_t><<<blocks, threads>>>(
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