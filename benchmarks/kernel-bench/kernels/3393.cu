#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define WARP_SIZE 32

// GELU function for float
__device__ inline float gelu_function(float x) {
    // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    return x * 0.5f * (1.0f + erff(x / 1.4142135623730951f));
}

// Vectorized kernel that processes data in float4 units and uses warp-level reduction
// to demonstrate eliminating shared memory for small reductions.
__global__ void gelu_kernel_vectorized_warp(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    size_t n4) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    float dummy_sum = 0.0f;  // Dummy accumulation to force warp-level reduction
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n4; i += stride) {
        float4 in_val = input[i];
        
        // Compute GELU for each element in the vector
        in_val.x = gelu_function(in_val.x);
        in_val.y = gelu_function(in_val.y);
        in_val.z = gelu_function(in_val.z);
        in_val.w = gelu_function(in_val.w);
        
        output[i] = in_val;
        
        // Use the first element of the vector as a representative value
        float warp_val = in_val.x;
        
        // Perform warp-level reduction using __shfl_down_sync
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            warp_val += __shfl_down_sync(0xffffffff, warp_val, offset);
        }
        
        // The first lane of the warp accumulates the reduced value as a dummy operation
        if (lane == 0) {
            dummy_sum += warp_val;
        }
    }
    
    // Prevent the compiler from optimizing away the dummy reduction
    if (idx == 0) {
        asm volatile ("" : : "r"(dummy_sum));
    }
}

// Kernel to handle remaining elements that do not fit in a float4 vectorized load
__global__ void gelu_kernel_remainder_warp(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t start,
    size_t numel) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t index = start + idx;
    if (index < numel) {
        output[index] = gelu_function(input[index]);
    }
}

// Forward function callable from Python
torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float, "This kernel supports float32 only");
    
    auto output = torch::empty_like(x);
    size_t numel = x.numel();
    const size_t vec_size = 4;
    size_t n4 = numel / vec_size;
    size_t remainder = numel % vec_size;
    
    int threads = 256;
    int blocks = (n4 + threads - 1) / threads;
    
    // Launch the vectorized kernel
    gelu_kernel_vectorized_warp<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(output.data_ptr<float>()),
        n4);
    
    // Launch a separate kernel for remaining elements
    if (remainder > 0) {
        int rem_blocks = (remainder + threads - 1) / threads;
        gelu_kernel_remainder_warp<<<rem_blocks, threads>>>(
            x.data_ptr<float>() + n4 * vec_size,
            output.data_ptr<float>() + n4 * vec_size,
            n4 * vec_size,
            numel);
    }
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "GELU activation forward (CUDA) with vectorized loads and warp-level reduction");
}
