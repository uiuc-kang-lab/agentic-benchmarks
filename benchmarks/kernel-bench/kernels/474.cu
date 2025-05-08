#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void coalescedMultiplyKernel(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       float s,
                                       int64_t size)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int aligned_size = (size + 3) & ~3;  // Round up to multiple of 4
    
    int base_idx = warp_id * 128 + lane_id * 4;  // Each warp processes 128 elements
    
    while (base_idx < aligned_size) {
        if (base_idx + 3 < size) {
            float4 elements;
            float4* a4_ptr = (float4*)(&A[base_idx]);
            float4* c4_ptr = (float4*)(&C[base_idx]);
            
            elements = *a4_ptr;
            elements.x *= s;
            elements.y *= s;
            elements.z *= s;
            elements.w *= s;
            
            *c4_ptr = elements;
        }
        else {
            for (int i = 0; i < 4 && base_idx + i < size; i++) {
                C[base_idx + i] = A[base_idx + i] * s;
            }
        }
        
        base_idx += blockDim.x * gridDim.x * 4;
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    const int threads = 256;
    const int blocks = min(256, (int)((size + threads * 4 - 1) / (threads * 4)));
    
    coalescedMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                C.data_ptr<float>(),
                                                s,
                                                size);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced matrix-scalar multiplication kernel");
}