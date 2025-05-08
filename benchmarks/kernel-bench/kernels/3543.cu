#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Constants for SELU activation
constexpr float ALPHA = 1.67326324235437728481f;
constexpr float LAMBDA = 1.05070098735548049342f;

__device__ __forceinline__ float my_exp(float x) {
    return expf(x);
}

__device__ __forceinline__ void process_vector(const float4& in_vec, float4& out_vec) {
    // Process each element of the vector
    out_vec.x = (in_vec.x > 0.0f) ? in_vec.x : ALPHA * (my_exp(in_vec.x) - 1.0f);
    out_vec.y = (in_vec.y > 0.0f) ? in_vec.y : ALPHA * (my_exp(in_vec.y) - 1.0f);
    out_vec.z = (in_vec.z > 0.0f) ? in_vec.z : ALPHA * (my_exp(in_vec.z) - 1.0f);
    out_vec.w = (in_vec.w > 0.0f) ? in_vec.w : ALPHA * (my_exp(in_vec.w) - 1.0f);
    
    // Apply lambda scaling
    out_vec.x *= LAMBDA;
    out_vec.y *= LAMBDA;
    out_vec.z *= LAMBDA;
    out_vec.w *= LAMBDA;
}

__global__ void selu_kernel_warp_optimized(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         const size_t numel) {
    // Shared memory for cooperative loading
    __shared__ float4 shared_data[32];
    
    const unsigned int tid = threadIdx.x;
    const unsigned int lane_id = tid & 0x1F;
    const unsigned int warp_id = tid >> 5;
    const size_t global_idx = (blockIdx.x * blockDim.x + tid) * 4;
    
    // Each thread processes 4 elements at a time using float4
    if (global_idx < numel) {
        // Load using vectorized reads
        float4 in_vec = reinterpret_cast<const float4*>(input)[global_idx / 4];
        float4 out_vec;
        
        // Process the vector
        process_vector(in_vec, out_vec);
        
        // Warp-level reduction for any shared processing if needed
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            float4 temp;
            temp.x = __shfl_down_sync(0xFFFFFFFF, out_vec.x, offset);
            temp.y = __shfl_down_sync(0xFFFFFFFF, out_vec.y, offset);
            temp.z = __shfl_down_sync(0xFFFFFFFF, out_vec.z, offset);
            temp.w = __shfl_down_sync(0xFFFFFFFF, out_vec.w, offset);
            
            if (lane_id < offset) {
                // Combine results if needed (in this case, just store)
                shared_data[lane_id] = out_vec;
            }
        }
        
        // Store results back to global memory
        if (global_idx + 3 < numel) {
            reinterpret_cast<float4*>(output)[global_idx / 4] = out_vec;
        } else {
            // Handle edge cases
            const size_t remaining = numel - global_idx;
            float* scalar_output = &output[global_idx];
            const float results[4] = {out_vec.x, out_vec.y, out_vec.z, out_vec.w};
            for (size_t i = 0; i < remaining; ++i) {
                scalar_output[i] = results[i];
            }
        }
    }
}

torch::Tensor selu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Input must be float32");
    
    auto output = torch::empty_like(input);
    const size_t numel = input.numel();
    
    // Configure kernel launch parameters
    const int threads = 256;  // Multiple of 32 for optimal warp utilization
    const int vector_elements = 4;
    const int blocks = (numel + (threads * vector_elements) - 1) / (threads * vector_elements);
    
    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    
    // Launch kernel with shared memory size for float4 data
    const size_t shared_mem_size = 32 * sizeof(float4);
    selu_kernel_warp_optimized<<<blocks, threads, shared_mem_size>>>(
        input_ptr, output_ptr, numel);
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &selu_forward, "SELU Activation Forward (CUDA) with Warp Optimization");
}