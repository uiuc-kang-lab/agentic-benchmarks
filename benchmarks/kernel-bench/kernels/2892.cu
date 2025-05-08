#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory for lookup table
__constant__ float c_exp_lookup[256];
__constant__ bool c_initialized = false;

// Host-side initialization of constant memory
void initialize_constant_memory() {
    float host_exp_lookup[256];
    for(int i = 0; i < 256; i++) {
        // Pre-compute exp values for common ranges
        float x = (i - 128) / 16.0f;  // Maps to [-8, 8] range
        host_exp_lookup[i] = expf(-x);
    }
    cudaMemcpyToSymbol(c_exp_lookup, host_exp_lookup, 256 * sizeof(float));
    bool init = true;
    cudaMemcpyToSymbol(c_initialized, &init, sizeof(bool));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t fast_sigmoid(scalar_t x) {
    // Fast path using constant memory lookup for common ranges
    if (x >= -8.0f && x <= 8.0f) {
        float normalized = (x + 8.0f) * 16.0f;
        int idx = min(255, max(0, int(normalized)));
        return scalar_t(1.0f / (1.0f + c_exp_lookup[idx]));
    }
    // Fallback for values outside lookup range
    return scalar_t(1.0f / (1.0f + expf(-x)));
}

template <typename scalar_t>
__global__ void sigmoid_kernel_const(const scalar_t* __restrict__ input,
                                   scalar_t* __restrict__ output,
                                   const int64_t size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements per thread when possible
    const int vec_size = 4;
    const int vec_elements = size / vec_size;
    
    for(int i = idx; i < vec_elements; i += stride) {
        float4 in_val = reinterpret_cast<const float4*>(input)[i];
        float4 out_val;
        out_val.x = fast_sigmoid(in_val.x);
        out_val.y = fast_sigmoid(in_val.y);
        out_val.z = fast_sigmoid(in_val.z);
        out_val.w = fast_sigmoid(in_val.w);
        reinterpret_cast<float4*>(output)[i] = out_val;
    }
    
    // Handle remaining elements
    for(int i = vec_elements * vec_size + idx; i < size; i += stride) {
        output[i] = fast_sigmoid(input[i]);
    }
}

torch::Tensor forward(torch::Tensor input) {
    static bool initialized = false;
    if (!initialized) {
        initialize_constant_memory();
        initialized = true;
    }
    
    auto output = torch::empty_like(input);
    const int64_t size = input.numel();
    
    const int threads = 256;
    const int blocks = (size + threads * 4 - 1) / (threads * 4);
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "sigmoid_kernel_const", [&] {
        sigmoid_kernel_const<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size
        );
    });
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Sigmoid forward with constant memory (CUDA)");
}