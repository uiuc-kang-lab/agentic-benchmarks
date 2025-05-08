#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function for GELU activation
__device__ __forceinline__ float gelu_activation(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Kernel that uses vectorized memory accesses for coalesced global memory reads/writes
__global__ void gelu_kernel_coalesced(const float* __restrict__ x, float* __restrict__ y, int n) {
    // Calculate overall thread index and total number of threads
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    // Process the bulk of the data as float4 vectors for coalesced memory accesses
    int n_vec = n / 4;  // number of float4 elements
    int remainder = n % 4;
    
    // Cast pointers to float4 to perform vectorized loads and stores
    const float4* x_vec = reinterpret_cast<const float4*>(x);
    float4* y_vec = reinterpret_cast<float4*>(y);
    
    for (int i = idx; i < n_vec; i += total_threads) {
        float4 in_val = x_vec[i];
        float4 out_val;
        out_val.x = gelu_activation(in_val.x);
        out_val.y = gelu_activation(in_val.y);
        out_val.z = gelu_activation(in_val.z);
        out_val.w = gelu_activation(in_val.w);
        y_vec[i] = out_val;
    }
    
    // Process any remaining elements that do not form a complete float4
    int base = n_vec * 4;
    for (int i = idx; i < remainder; i += total_threads) {
        int index = base + i;
        y[index] = gelu_activation(x[index]);
    }
}

// Host function to launch the kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    auto y = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    // Compute number of blocks based on the number of vectorized (float4) elements
    int n_vec = n / 4;
    int blocks = (n_vec + threads - 1) / threads;
    if (blocks < 1) blocks = 1; // Ensure at least one block is launched

    gelu_kernel_coalesced<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Coalesced vectorized GELU forward CUDA implementation");
}
