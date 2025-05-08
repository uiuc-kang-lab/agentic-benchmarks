#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Vectorized kernel with shared memory for large chunks
__global__ void elu_kernel_vec4_shared(const float4* x, float4* out, float alpha, int n4) {
    extern __shared__ float4 tile[];
    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + tid;

    // Load input data using vectorized reads
    if (globalIdx < n4) {
        tile[tid] = x[globalIdx];
    }
    __syncthreads();

    if (globalIdx < n4) {
        float4 val = tile[tid];
        float4 result;
        
        result.x = (val.x > 0) ? val.x : alpha * (expf(val.x) - 1);
        result.y = (val.y > 0) ? val.y : alpha * (expf(val.y) - 1);
        result.z = (val.z > 0) ? val.z : alpha * (expf(val.z) - 1);
        result.w = (val.w > 0) ? val.w : alpha * (expf(val.w) - 1);
        
        out[globalIdx] = result;
    }
}

// Regular kernel for remaining elements
__global__ void elu_kernel_remainder(const float* x, float* out, float alpha, int start, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + start < n) {
        float val = x[idx + start];
        out[idx + start] = (val > 0) ? val : alpha * (expf(val) - 1);
    }
}

torch::Tensor elu_cuda_optimal(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    
    int n = x.numel();
    int n4 = n / 4;  // Number of float4 elements
    int remainder = n % 4;  // Remaining elements
    
    // Experiment with different block sizes
    const int threads = 512;  // Optimal block size for the given hardware
    const int blocks = (n4 + threads - 1) / threads;
    
    // Process main part using vectorized loads and shared memory
    if (n4 > 0) {
        size_t sharedMemSize = threads * sizeof(float4);
        elu_kernel_vec4_shared<<<blocks, threads, sharedMemSize>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            alpha,
            n4
        );
    }
    
    // Process remaining elements
    if (remainder > 0) {
        const int remainder_blocks = (remainder + threads - 1) / threads;
        elu_kernel_remainder<<<remainder_blocks, threads>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            alpha,
            n4 * 4,
            n
        );
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_optimal, "Optimal ELU activation (CUDA)");
}