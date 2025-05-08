#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__constant__ float c_scalar;

__global__ void multiplySharedMemoryKernel(const float* __restrict__ A,
                                           float* __restrict__ C,
                                           int64_t size) {
    extern __shared__ float4 smem[];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    
    // Load 4 float4 elements per thread to shared memory
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        int load_idx = tid + i * blockDim.x * gridDim.x;
        if(load_idx * 4 < size) {
            smem[threadIdx.x + i * blockDim.x] = 
                reinterpret_cast<const float4*>(A)[load_idx];
        }
    }
    __syncthreads();

    // Process data from shared memory with warp-level optimization
    float val = ((float*)smem)[threadIdx.x * 4];
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        float element = __shfl_down_sync(0xffffffff, val, i);
        if(lane_id + i < 32) {
            ((float*)smem)[threadIdx.x * 4 + i] = element * c_scalar;
        }
    }
    __syncthreads();

    // Store results back to global memory
    #pragma unroll
    for(int i = 0; i < 4; ++i) {
        int store_idx = tid + i * blockDim.x * gridDim.x;
        if(store_idx * 4 < size) {
            reinterpret_cast<float4*>(C)[store_idx] = 
                smem[threadIdx.x + i * blockDim.x];
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    // Configure kernel
    const int threads = 256;
    const int blocks = (size + threads * 4 - 1) / (threads * 4);
    
    // Copy scalar to constant memory
    cudaMemcpyToSymbol(c_scalar, &s, sizeof(float));
    
    // Launch kernel with shared memory allocation
    multiplySharedMemoryKernel<<<blocks, threads, threads * sizeof(float4) * 4>>>(
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        size
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication with shared memory and warp optimization");
}